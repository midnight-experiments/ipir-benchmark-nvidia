/**
 * Shared pieces of the GPU benchmarks: opening the adapter (and refusing a software one), synthetic tables and
 * requests as the proof of concept's timing harness makes them, and statistics.
 *
 * The synthetic inputs: cells from xorshift32, random 32-bit packing matrices (2,048 × 4,096 per column block),
 * requests with the right magic and a fixed setup id and random words. `GpuIpirTable`'s time does not depend on the
 * content, only on the shape.
 */
import { BLOCK_BUILD_LIMITS, blockBuildSupported } from "../gpu/block-build.ts";
import { ipirRequestBytes } from "../ipir/sizes.ts";

export const MIB = 2 ** 20;
export const GIB = 2 ** 30;
/** Bytes of one block's compiled packing matrix on the GPU (2,048 × 4,096 × 4). */
export const MATRIX_BYTES = 2048 * 4096 * 4;

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean | null;
  features: string[];
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  timestampQuery: boolean;
  /** Whether the device was opened with the limits the GPU block build needs (`gpu/block-build.ts`). */
  blockBuild: boolean;
  /** A software rasterizer (llvmpipe, lavapipe, SwiftShader, WARP, a fallback adapter): its numbers mean nothing. */
  software: boolean;
  softwareReason: string | null;
}

export interface BenchGpu {
  device: GPUDevice;
  adapter: AdapterInfo;
  /** As `openGpu`'s description (vendor, architecture, device, description). */
  description: string;
}

const SOFTWARE = /llvmpipe|lavapipe|softpipe|swiftshader|software|basic render|\bwarp\b|\bcpu\b/i;

/**
 * Opens the adapter as the proof of concept's `openGpu` does (`gpu/device.ts`): high-performance, the adapter's
 * largest buffer limits, `timestamp-query` when the adapter has it. Also reports the adapter's details. Throws when
 * there is no WebGPU or no adapter.
 */
export async function openBenchGpu(): Promise<BenchGpu> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) throw new Error("this Deno has no WebGPU (navigator.gpu is missing)");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter found (see the README, Troubleshooting: no adapter)");
  const info = adapter.info as GPUAdapterInfo & { isFallbackAdapter?: boolean };
  const features = [...adapter.features].map(String).sort();
  const text = [info.vendor, info.architecture, info.device, info.description].join(" ");
  const fallback = info.isFallbackAdapter ?? null;
  const softwareMatch = text.match(SOFTWARE);
  const software = fallback === true || softwareMatch !== null;
  const blockBuild = blockBuildSupported(adapter.limits);
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      // The GPU block build's kernels: 1,024 invocations per workgroup and 32 KB of workgroup memory.
      ...(blockBuild ? BLOCK_BUILD_LIMITS : {}),
    },
  });
  const description = [info.vendor, info.architecture, info.device, info.description]
    .filter((s) => s && s !== "0")
    .join(" ")
    .trim() || "unknown adapter";
  return {
    device,
    description,
    adapter: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
      isFallbackAdapter: fallback,
      features,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      timestampQuery: device.features.has("timestamp-query"),
      blockBuild,
      software,
      softwareReason: software ? (fallback ? "fallback adapter" : `adapter text matches "${softwareMatch![0]}"`) : null,
    },
  };
}

export function xorshift(seed: number): () => number {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x;
  };
}

/** `rows × cols` column-major 16-bit cells. */
export function syntheticCells(rows: number, cols: number, next: () => number): Uint16Array {
  const cells = new Uint16Array(rows * cols);
  for (let i = 0; i < cells.length; i++) cells[i] = next() & 0xffff;
  return cells;
}

/** One block's packing matrix: 2,048 × 4,096 signed entries, as the harness made them (below 2^25 in magnitude). */
export function syntheticMatrix(next: () => number): Int32Array {
  const m = new Int32Array(2048 * 4096);
  for (let i = 0; i < m.length; i++) m[i] = (next() % (1 << 26)) - (1 << 25);
  return m;
}

export function syntheticSetupId(next: () => number): Uint8Array {
  return new Uint8Array(32).map(() => next() & 0xff);
}

/** `count` requests laid end to end, each with the magic "RNQ3" and `setupId`, the rest random. */
export function syntheticRequests(rows: number, count: number, setupId: Uint8Array, next: () => number): Uint8Array {
  const each = ipirRequestBytes(rows);
  const requests = new Uint8Array(count * each);
  for (let i = 0; i < requests.length; i++) requests[i] = next() & 0xff;
  const magic = new TextEncoder().encode("RNQ3");
  for (let i = 0; i < count; i++) {
    requests.set(magic, i * each);
    requests.set(setupId, i * each + 4);
  }
  return requests;
}

/** Bytes a table keeps on the GPU: 2 per cell and 32 MiB per column block. */
export function residentBytes(rows: number, cols: number): number {
  return rows * cols * 2 + (cols / 2048) * MATRIX_BYTES;
}

/** Nearest-rank percentile (p in 0..100) of the values; null when there are none. */
export function percentile(xs: readonly number[], p: number): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const rank = Math.min(s.length, Math.max(1, Math.ceil((p / 100) * s.length)));
  return s[rank - 1]!;
}

export function median(xs: readonly number[]): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2;
}

export function stats(xs: readonly number[]) {
  const round = (x: number | null) => (x === null ? null : Math.round(x * 1000) / 1000);
  return {
    n: xs.length,
    min: round(xs.length ? Math.min(...xs) : null),
    p50: round(percentile(xs, 50)),
    p90: round(percentile(xs, 90)),
    p99: round(percentile(xs, 99)),
    max: round(xs.length ? Math.max(...xs) : null),
  };
}

/** Parses `--name value` and `--flag` options. */
export function options(args: readonly string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    if (!a.startsWith("--")) throw new Error(`unexpected argument ${a}`);
    const next = args[i + 1];
    if (next === undefined || next.startsWith("--")) out.set(a.slice(2), "true");
    else {
      out.set(a.slice(2), next);
      i++;
    }
  }
  return out;
}

export function parseShapes(s: string): [number, number][] {
  return s.split(",").filter(Boolean).map((x) => {
    const [r, c] = x.toLowerCase().split("x").map(Number);
    if (!r || !c) throw new Error(`bad shape ${x} (expected ROWSxCOLS)`);
    return [r, c];
  });
}

/**
 * Runs `f` inside out-of-memory and validation error scopes and throws if either caught an error (WebGPU reports a
 * failed allocation there, not as an exception).
 */
export async function scoped<T>(device: GPUDevice, what: string, f: () => Promise<T>): Promise<T> {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  let value: T | undefined;
  let thrown: unknown = null;
  try {
    value = await f();
  } catch (e) {
    thrown = e;
  }
  const internal = await device.popErrorScope();
  const validation = await device.popErrorScope();
  const oom = await device.popErrorScope();
  if (oom) throw new Error(`${what}: GPU out of memory: ${oom.message}`);
  if (validation) throw new Error(`${what}: GPU validation error: ${validation.message}`);
  if (internal) throw new Error(`${what}: GPU internal error: ${internal.message}`);
  if (thrown) throw thrown;
  return value as T;
}
