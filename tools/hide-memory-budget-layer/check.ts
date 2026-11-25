/**
 * Creates one GPU buffer on a fresh WebGPU device and reports whether wgpu allowed it.
 *
 *   deno run -A tools/hide-memory-budget-layer/check.ts [MiB, default 128]
 *
 * Run it without and then with the layer (see README.md). 128 MiB is the largest buffer the benchmark's
 * 2048 × 8192 table needs.
 */
const mib = Number(Deno.args[0] ?? 128);
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no WebGPU adapter found");
const device = await adapter.requestDevice({
  requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize },
});
let lost = "";
device.lost.then((info) => (lost = `${info.reason}: ${info.message}`));
device.pushErrorScope("out-of-memory");
const buffer = device.createBuffer({ size: mib * 2 ** 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const error = await device.popErrorScope();
await new Promise((resolve) => setTimeout(resolve, 200));
console.log(`${mib} MiB: ${error ? `refused (${error.message})` : "ok"}${lost ? `; device lost (${lost})` : ""}`);
buffer.destroy();
device.destroy();
