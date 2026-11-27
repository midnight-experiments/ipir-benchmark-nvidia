# Hosts: what a machine needs

What `./bench.sh` needs on the machine it measures, how to prepare a rented Linux machine, and what the CUDA
measurements need.

## Host requirements

On macOS (Apple silicon), only Rust and `curl`/`unzip` are needed: `setup.sh` installs Deno for darwin-aarch64, and
Deno's WebGPU uses Metal. There is no `nvidia-smi` or `vulkaninfo` to check there. The rest of this section is for
Linux with an NVIDIA GPU.

- Linux, x86_64: Ubuntu 22.04 or 24.04, or similar. Root is needed only to install the driver and the packages below.
- NVIDIA driver **550 or newer** (535 should work; the RTX 2060 is Turing). The driver must include its **Vulkan ICD**:
  on Ubuntu that is `libnvidia-gl-<branch>`, which the desktop driver packages pull in. The headless/server packages
  (`nvidia-headless-*`) do not, so install `libnvidia-gl-<branch>` too.
  Check: `ls /usr/share/vulkan/icd.d/nvidia_icd.json`.
- Vulkan loader and tools: `sudo apt install libvulkan1 vulkan-tools`. `vulkaninfo --summary` must list the RTX 2060
  as `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`.
- Rust through rustup (https://rustup.rs). `rust-toolchain.toml` pins **1.92.0**, which rustup installs on first use.
  Also a C toolchain: `sudo apt install build-essential`.
- Deno **2.7.5**. `./setup.sh` downloads it into `.tools/deno/`, so a system Deno is not needed. WebGPU is enabled by
  default in this version; no `--unstable-webgpu` flag is needed. On Linux, Deno 2.7.5's WebGPU is wgpu 28 on
  Vulkan, and `run-all.sh` sets `DENO_WEBGPU_BACKEND=vulkan`.
- `curl`, `unzip`, `git`. About 16 GB RAM (the largest shape makes 2.5 GiB of cells and matrices on the host), 3 GB
  of disk for the Rust build.
- A machine that is otherwise idle. Close other GPU programs: `nvidia-smi` should show no other compute processes,
  and little memory in use by a desktop.

## A rented Linux machine with resizable BAR (the `nvidia` backend), and CUDA

Rent a VM with one recent NVIDIA GPU passed through whole (an L4, L40S, A10, A100, H100 or an RTX 4090/5090 class
card; not a vGPU or MIG slice), 8 or more vCPUs, 32 GB of RAM and Ubuntu 22.04 or 24.04. Data-centre GPUs and most
recent clouds expose the full BAR (a T4 does not: its BAR1 is 256 MiB). A whole VM is simplest: the driver and its
Vulkan part are yours to install.

**A container instead of a VM** (RunPod, vast.ai and similar rent containers): the host's driver is mounted into
the container, and only the parts its `NVIDIA_DRIVER_CAPABILITIES` names. The default, `compute,utility`, leaves out
the Vulkan driver (`libGLX_nvidia.so.0`), and WebGPU then finds no NVIDIA device. Start the container with
`NVIDIA_DRIVER_CAPABILITIES=all` (an environment variable in the provider's template; with Docker,
`docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ...`). Do not install a driver inside it. If
`libGLX_nvidia.so.0` is there but `vulkaninfo` still sees no NVIDIA device, the ICD file is missing; `./setup.sh`
prints the one line that creates it.

Then:

1. On a VM, the driver with its Vulkan part, which `bench.sh` does not install:
   `sudo apt install nvidia-driver-570-server libnvidia-gl-570-server` (or the `-open` and desktop branches;
   whatever the image has, add its `libnvidia-gl-<branch>`), then reboot. `nvidia-smi` must list the GPU.
2. Clone and run the quick check first:
   ```sh
   git clone https://github.com/midnight-experiments/ipir-benchmark-nvidia.git && cd ipir-benchmark-nvidia
   ./bench.sh --install-deps --quick --host <short name>
   ```
   What to look for in its output:
   - `GPU backend: nvidia`, and `bar1 total` at least `fb total` (for example 32768 MiB BAR1 for a 24 GB card). If
     BAR1 is 256 MiB, resizable BAR is off, and the run would be `nvidia-no-bar`.
   - `without the layer, 1024 MiB: ok`: wgpu allows the buffers, and the layer is not loaded.
   - The WebGPU adapter is the NVIDIA GPU, not llvmpipe.
   - It ends with `The quick check passed`.
3. The full run, on an otherwise idle machine, inside `tmux`:
   ```sh
   ./bench.sh --skip-quick --host <short name> --label "<GPU> + <CPU>"          # the nvidia file, about 100 minutes
   ./bench.sh --skip-quick --host <short name> --label "<GPU> + <CPU>" --cpu --cuda   # or with that host's CPU and cuda_bench
   ```
   Then send back `results/<short name>-<date>-bundle.tar.gz`, or push (the README's run instructions).
4. `--cuda` measures upstream's CUDA kernel (`simplepir-kernel`, `CudaKernel`) against the WebGPU path. The kernel
   does the first pass of an answer on the GPU, one request at a time, and nothing else, so the comparison is a
   whole answer two ways (`bench/cuda-compare.ts`, in the full run, step 4 of its summary):
   - CUDA: the first pass on the GPU and the packing of every column block on the host's CPU (`cuda_fetch`);
   - WebGPU: both steps on the GPU;
   - for the same table and the same 32 requests, with every answer compared with the CPU's, byte for byte.

   `./bench.sh --report --cuda` also measures this repository's own CUDA path, which upstream's kernel is compared
   with in the report:
   - `cuda_fetch_native`: a whole answer with the first pass for all the requests of a call in one launch and the
     packing on the GPU. Before any timing it checks its answers, its first pass and its packing product against the
     CPU's, and a column block replaced from the host and from the CUDA block build. `--sweep` times the layouts a
     call can have on the card and prints the fastest as a value for `IPIR_CUDA_FETCH`; the defaults are the
     fastest on the RTX 4090 and within 5 % of it on the RTX 2060.
   - `cuda_budget`: one chain block built on the GPU and installed into six tables while they answer 2 calls a
     second.

   It also times the upload of the table again, which is what a changed column block costs on the CUDA kernel,
   against WebGPU's write of one block. `bench.sh --cuda` then runs upstream's own `cuda_bench` (one request's
   first pass with another modulus; `results/<short name>-<date>-cuda-bench.txt`).

   CUDA needs the driver library (`libcuda.so`, part of the NVIDIA driver) and NVRTC (`libnvrtc.so`, part of the
   CUDA toolkit), loaded at run time:
   - The bindings are built for CUDA 13 when `libnvrtc.so.13` is installed and for CUDA 12 otherwise
     (`IPIR_CUDA_FEATURE=cuda-12` or `cuda-13` chooses outright). Upstream fixes CUDA 12.0, whose bindings do not
     load with the NVRTC of CUDA 13 (`undefined symbol: nvrtcGetNVVM`); patch 0004 makes it a choice.
   - NVRTC must not be newer than the driver: `nvidia-smi` prints the CUDA release the driver supports, and an NVRTC
     of a later release compiles kernels the driver refuses (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).

   By hand, on a results folder that exists:
   ```sh
   (cd ipir && cargo build --release --locked -p ipir-bench --features cuda-13 --bin cuda_fetch --bin cuda_fetch_native --bin cuda_budget)
   ipir/target/release/cuda_fetch_native 14336 34816 9 --throughput 3     # the checks, the times, the calls per second
   deno run --allow-read --allow-write --allow-run --allow-env --allow-sys bench/cuda-compare.ts --out results/raw/<run>/cuda-compare.json
   deno run --allow-read --allow-write --allow-run --allow-env --allow-sys bench/report.ts results/raw/<run> results/<run>.json results/<run>.md
   ```

The results record: GPU name, driver, VRAM and VRAM in use (`nvidia-smi`); the Vulkan devices (`vulkaninfo
--summary`); the WebGPU adapter Deno picked, and whether it has timestamp queries; CPU model, cores and threads; RAM;
OS and kernel; Deno and Rust versions; the load average at the start and the end.
