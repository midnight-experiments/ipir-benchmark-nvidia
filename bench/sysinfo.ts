/**
 * The machine, for the results file: GPU name, driver and memory (nvidia-smi), the Vulkan device list (vulkaninfo),
 * the WebGPU adapter Deno sees, CPU model and cores, RAM, OS, Deno and Rust versions, and the load average.
 *
 * Also the GPU backend this host's answers run on (`gpuBackend`): `metal` on macOS; on Linux with an NVIDIA GPU,
 * `nvidia` when resizable BAR is on and `nvidia-no-bar` when it is off. Resizable BAR is decided from `nvidia-smi -q
 * -d MEMORY`: on when the BAR1 total is at least the framebuffer total (`nvidiaMemory` keeps the raw values). And
 * whether the Vulkan layer in `tools/hide-memory-budget-layer/` is loaded (`memoryBudgetLayer`, from
 * `VK_INSTANCE_LAYERS`).
 *
 *   deno run <permissions> bench/sysinfo.ts [--no-adapter]
 *
 * Prints one JSON object. Exits 3 when the adapter is a software one (llvmpipe, lavapipe, SwiftShader, a fallback
 * adapter): the benchmark must not run on it.
 */
import { openBenchGpu, options } from "./lib.ts";

const opts = options(Deno.args);

async function run(cmd: string, args: string[]): Promise<string | null> {
  try {
    const r = await new Deno.Command(cmd, { args, stdout: "piped", stderr: "piped" }).output();
    if (!r.success) return null;
    return new TextDecoder().decode(r.stdout).trim();
  } catch {
    return null;
  }
}
/** A file's text; through `cat` when Deno refuses the read (Deno 2 refuses `/proc` and `/sys` without --allow-all). */
async function read(path: string): Promise<string | null> {
  try {
    return await Deno.readTextFile(path);
  } catch {
    return await run("cat", [path]);
  }
}

const os = Deno.build.os;
const info: Record<string, unknown> = {
  date: new Date().toISOString(),
  hostname: Deno.env.get("IPIR_BENCH_HOST") || Deno.hostname(),
  os,
  arch: Deno.build.arch,
  osRelease: Deno.osRelease(),
  loadavg: Deno.loadavg(),
  deno: Deno.version,
  env: Object.fromEntries(
    ["DENO_WEBGPU_BACKEND", "VK_INSTANCE_LAYERS", "VK_LAYER_PATH", "VK_DRIVER_FILES", "VK_ICD_FILENAMES"].map((k) => [k, Deno.env.get(k) ?? null]),
  ),
};
const layers = (Deno.env.get("VK_INSTANCE_LAYERS") ?? "").split(":");
info.memoryBudgetLayer = layers.includes("VK_LAYER_IPIRBENCH_hide_memory_budget");

// GPU: nvidia-smi.
const smiFields = [
  "name", "driver_version", "memory.total", "memory.used", "memory.free", "pcie.link.gen.current",
  "pcie.link.width.current", "clocks.max.sm", "clocks.max.mem", "power.limit", "temperature.gpu", "pstate",
  "compute_mode", "vbios_version",
];
const smi = await run("nvidia-smi", [`--query-gpu=${smiFields.join(",")}`, "--format=csv,noheader,nounits"]);
info.nvidiaSmi = smi === null ? null : smi.split("\n").map((line) => {
  const vals = line.split(",").map((s) => s.trim());
  return Object.fromEntries(smiFields.map((f, i) => [f, vals[i] ?? null]));
});
info.nvidiaSmiVersion = (await run("nvidia-smi", ["--version"]))?.split("\n").filter(Boolean) ?? null;
info.gpuProcesses = await run("nvidia-smi", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]);
// Resizable BAR: with it, the BAR1 aperture covers the whole framebuffer (BAR1 total >= FB total); without it, BAR1 is
// a small window (256 MiB on most cards).
const smiMemory = await run("nvidia-smi", ["-q", "-d", "MEMORY"]);
if (smiMemory !== null) {
  const total = (section: string) => {
    const m = smiMemory.match(new RegExp(`${section}[^\\n]*\\n\\s*Total\\s*:\\s*(\\d+)\\s*MiB`));
    return m ? Number(m[1]) : null;
  };
  const fbTotalMiB = total("FB Memory Usage");
  const bar1TotalMiB = total("BAR1 Memory Usage");
  const bar1Used = smiMemory.match(/BAR1 Memory Usage[\s\S]*?Used\s*:\s*(\d+)\s*MiB/);
  info.nvidiaMemory = {
    fbTotalMiB,
    bar1TotalMiB,
    bar1UsedMiB: bar1Used ? Number(bar1Used[1]) : null,
    resizableBar: fbTotalMiB !== null && bar1TotalMiB !== null ? bar1TotalMiB >= fbTotalMiB : null,
    raw: smiMemory,
  };
} else info.nvidiaMemory = null;
const vk = await run("vulkaninfo", ["--summary"]);
if (vk !== null) {
  const devices: Record<string, string>[] = [];
  let cur: Record<string, string> | null = null;
  for (const line of vk.split("\n")) {
    if (/^GPU\d+:/.test(line.trim())) {
      cur = {};
      devices.push(cur);
      continue;
    }
    const m = line.match(/^\s+(apiVersion|driverVersion|vendorID|deviceID|deviceType|deviceName|driverID|driverName|driverInfo)\s*=\s*(.*)$/);
    if (m && cur) cur[m[1]!] = m[2]!.trim();
  }
  const instance = vk.match(/Vulkan Instance Version:\s*(\S+)/)?.[1] ?? null;
  info.vulkan = { instanceVersion: instance, devices };
} else info.vulkan = null;

// CPU and memory.
if (os === "linux") {
  const cpuinfo = await read("/proc/cpuinfo");
  info.cpuModel = cpuinfo?.match(/model name\s*:\s*(.*)/)?.[1] ?? null;
  info.lscpu = await run("lscpu", []);
  const meminfo = await read("/proc/meminfo");
  info.memTotalKiB = Number(meminfo?.match(/MemTotal:\s*(\d+)/)?.[1] ?? NaN) || null;
  info.memAvailableKiB = Number(meminfo?.match(/MemAvailable:\s*(\d+)/)?.[1] ?? NaN) || null;
  info.cpuGovernor = (await read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))?.trim() ?? null;
  info.distro = (await read("/etc/os-release"))?.match(/PRETTY_NAME="?([^"\n]*)/)?.[1] ?? null;
} else if (os === "darwin") {
  info.cpuModel = await run("sysctl", ["-n", "machdep.cpu.brand_string"]);
  info.memTotalKiB = Number(await run("sysctl", ["-n", "hw.memsize"])) / 1024;
  info.perfCores = await run("sysctl", ["-n", "hw.perflevel0.physicalcpu"]);
  info.efficiencyCores = await run("sysctl", ["-n", "hw.perflevel1.physicalcpu"]);
  info.distro = `macOS ${await run("sw_vers", ["-productVersion"])}`;
  info.powerSource = (await run("pmset", ["-g", "batt"]))?.match(/'([^']+)'/)?.[1] ?? null;
  try {
    const sp = JSON.parse((await run("system_profiler", ["SPDisplaysDataType", "-json"])) ?? "{}");
    const g = sp.SPDisplaysDataType?.[0];
    info.macGpu = g ? { model: g.sppci_model ?? g._name ?? null, cores: g.sppci_cores ? Number(g.sppci_cores) : null, metal: g.spdisplays_mtlgpufamilysupport ?? null } : null;
  } catch {
    info.macGpu = null;
  }
}
info.logicalCpus = navigator.hardwareConcurrency;
info.physicalCpus = os === "linux"
  ? Number(await run("sh", ["-c", "lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l"])) || null
  : Number(await run("sysctl", ["-n", "hw.physicalcpu"])) || null;
info.rustc = await run("rustc", ["--version"]);
info.cargo = await run("cargo", ["--version"]);

const nvidiaMemory = info.nvidiaMemory as { resizableBar: boolean | null } | null;
info.gpuBackend = os === "darwin"
  ? "metal"
  : info.nvidiaSmi
  ? nvidiaMemory?.resizableBar === true ? "nvidia" : nvidiaMemory?.resizableBar === false ? "nvidia-no-bar" : "nvidia-unknown-bar"
  : null;

let software = false;
if (opts.get("no-adapter") !== "true") {
  try {
    const gpu = await openBenchGpu();
    info.webgpu = { summary: gpu.description, ...gpu.adapter };
    software = gpu.adapter.software;
    gpu.device.destroy();
  } catch (e) {
    info.webgpu = { error: String((e as Error).message ?? e) };
  }
}
console.log(JSON.stringify(info, null, 1));
if (software) Deno.exit(3);
