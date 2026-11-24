/**
 * The machine, for the results file: GPU name, driver and memory (nvidia-smi), the Vulkan device list (vulkaninfo),
 * the WebGPU adapter Deno sees, CPU model and cores, RAM, OS, Deno and Rust versions, and the load average.
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
async function read(path: string): Promise<string | null> {
  try {
    return await Deno.readTextFile(path);
  } catch {
    return null;
  }
}

const os = Deno.build.os;
const info: Record<string, unknown> = {
  date: new Date().toISOString(),
  hostname: Deno.hostname(),
  os,
  arch: Deno.build.arch,
  osRelease: Deno.osRelease(),
  loadavg: Deno.loadavg(),
  deno: Deno.version,
  env: { DENO_WEBGPU_BACKEND: Deno.env.get("DENO_WEBGPU_BACKEND") ?? null },
};

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
}
info.logicalCpus = navigator.hardwareConcurrency;
info.physicalCpus = os === "linux"
  ? Number(await run("sh", ["-c", "lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l"])) || null
  : Number(await run("sysctl", ["-n", "hw.physicalcpu"])) || null;
info.rustc = await run("rustc", ["--version"]);
info.cargo = await run("cargo", ["--version"]);

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
