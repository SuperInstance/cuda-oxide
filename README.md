<p align="center">
  <a href="https://github.com/SuperInstance/cuda-oxide/actions/workflows/clippy.yml"><img alt="clippy" src="https://github.com/SuperInstance/cuda-oxide/actions/workflows/clippy.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/SuperInstance/cuda-oxide/actions/workflows/unit-tests.yml"><img alt="unit-tests" src="https://github.com/SuperInstance/cuda-oxide/actions/workflows/unit-tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/SuperInstance/cuda-oxide/actions/workflows/cargo-deny.yml"><img alt="cargo-deny" src="https://github.com/SuperInstance/cuda-oxide/actions/workflows/cargo-deny.yml/badge.svg?branch=main"></a>
  <br>
  <img src="assets/logo.png" alt="cuda-oxide logo" width="100%">
</p>

# cuda-oxide

cuda-oxide is a custom rustc backend for compiling GPU kernels in pure Rust.
The workspace combines:

- single-source compilation -- host and device code live in the same file, built with one `cargo oxide build`
- a rustc codegen backend that compiles `#[kernel]` functions to CUDA PTX
- device-side abstractions (type-safe indexing, shared memory, scoped atomics, barriers, TMA, warp/cluster ops)
- a host-side runtime for memory management, pinned host transfers, and kernel launching (`cuda-core`, `cuda-async`)
- a rust-native compilation pipeline using [Pliron](https://github.com/vaivaswatha/pliron), an MLIR-like IR framework in Rust (Rust → Rust MIR → Pliron IR → LLVM IR → PTX)

## SuperInstance Fork

> This repository is **SuperInstance's fork** of the original [`NVlabs/cuda-oxide`](https://github.com/NVlabs/cuda-oxide) project.
>
> SuperInstance adopted cuda-oxide for **systems-level GPU development** — pushing the compiler beyond research demos into production-grade tooling for high-performance computing, agent runtimes, and bare-metal GPU orchestration. The fork maintains upstream compatibility while expanding architecture documentation, crate-level modularity, and long-term stability for systems workloads.
>
> **What changes in this fork:**
> - architecture documentation (`ARCHITECTURE.md`, `PIPELINE.md`) treating the 18-crate workspace as a compiler construction kit
> - Systems-focused runtime hardening (`cuda-core`, `cuda-async`) for async agent pipelines and memory-virtualization workloads
> - Crate-level READMEs for every workspace member, enabling selective reuse of individual pipeline stages
> - Fork-specific issue tracking and CI under the `SuperInstance` GitHub org

## Architecture Overview

The workspace is structured as **18 crates** across four layers:

| Layer | Crates | Purpose |
|-------|--------|---------|
| **Compiler backend** | `rustc-codegen-cuda`, `mir-importer`, `mir-lower`, `dialect-mir`, `dialect-nvvm`, `llvm-export` | Rust MIR → Pliron IR → LLVM IR → PTX |
| **Host runtime** | `cuda-core`, `cuda-host`, `cuda-async`, `cuda-bindings`, `libnvvm-sys`, `nvjitlink-sys` | Contexts, streams, buffers, async launches, LTOIR linking |
| **Device runtime** | `cuda-device`, `cuda-macros` | `#[kernel]`, `#[cuda_module]`, GPU intrinsics |
| **Build & infra** | `cargo-oxide`, `oxide-artifacts`, `reserved-oxide-symbols`, `fuzzer` | Cargo subcommand, embedded artifacts, differential fuzzing |

### Full Compilation Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RUST SOURCE (.rs)                                   │
│  #[cuda_module] mod kernels { #[kernel] fn add(...) { ... } }               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  rustc-codegen-cuda  ──  Custom rustc codegen backend (dylib)               │
│  • Splits host code → standard LLVM backend                                 │
│  • Extracts Stable MIR for #[kernel] fns → mir-importer                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│  HOST PATH (standard LLVM)      │   │  DEVICE PATH (cuda-oxide pipeline)  │
│  Host binary + embedded artifact│   │                                     │
└─────────────────────────────────┘   └─────────────────────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
                    ▼                               ▼                               ▼
┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
│   mir-importer           │  │   dialect-mir            │  │   dialect-nvvm           │
│   Rust MIR → dialect-mir │  │   Pliron MIR dialect     │  │   Pliron NVVM dialect    │
│   • translate_body()     │  │   • alloca/load/store    │  │   • thread/warp/atomic   │
│   • mem2reg (SSA promo)  │  │   • arithmetic, casts    │  │   • tma, wgmma, tcgen05  │
│   • verify_operation()   │  │   • enums, structs       │  │   • cluster, mbarrier    │
└──────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘
                    │                               │                               │
                    └───────────────────────────────┼───────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  mir-lower  ──  DialectConversion pass: dialect-mir → LLVM dialect           │
│  • flatten slice/struct args                                                │
│  • GPU intrinsics → NVVM calls / inline PTX asm                             │
│  • entry-block prologue reconstruction                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  llvm-export  ──  LLVM dialect → textual .ll (NVPTX)                         │
│  • datalayout + target triple                                               │
│  • block args → PHI nodes                                                   │
│  • @llvm.used + !nvvm.annotations                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  llc (LLVM 21+, -march=nvptx64)  ──  textual PTX assembly                   │
│  • Auto-detected SM target (sm_80 / sm_90 / sm_90a / sm_100a)               │
│  • Override: CUDA_OXIDE_TARGET=sm_100a                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOST BINARY  ──  oxide-artifacts embeds PTX → cuModuleLoad at runtime      │
│  • cuda-host generates typed module.map::<T, _>(...) launchers              │
│  • cuda-async provides DeviceOperation / DeviceFuture for composable GPU    │
└─────────────────────────────────────────────────────────────────────────────┘
```

For a step-by-step walkthrough of each compiler stage, see [`PIPELINE.md`](PIPELINE.md).
For deep-dive architecture docs (runtime, IR design, build system, data flow), see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Crate Reference

All 18 workspace crates with line counts and maturity status.

### Compiler Crates

| Crate | Role | LOC | Status |
|-------|------|-----|--------|
| [`rustc-codegen-cuda`](crates/rustc-codegen-cuda/README.md) | Custom rustc backend (dylib) — host/device split | ~2.7K | ✅ Active |
| [`mir-importer`](crates/mir-importer/README.md) | Rust MIR → `dialect-mir` translator + mem2reg + verify | ~23.9K | ✅ Active |
| [`mir-lower`](crates/mir-lower/README.md) | `dialect-mir` → LLVM dialect lowering (DialectConversion) | ~13.4K | ✅ Active |
| [`dialect-mir`](crates/dialect-mir/README.md) | Pliron dialect modelling Rust MIR semantics | ~6.1K | ✅ Active |
| [`dialect-nvvm`](crates/dialect-nvvm/README.md) | Pliron dialect modelling NVVM GPU intrinsics | ~5.5K | ✅ Active |
| [`llvm-export`](crates/llvm-export/README.md) | Pliron-LLVM shim + textual `.ll` exporter | ~3.0K | ✅ Active |

### Host Runtime Crates

| Crate | Role | LOC | Status |
|-------|------|-----|--------|
| [`cuda-core`](crates/cuda-core/README.md) | Safe RAII wrappers (`CudaContext`, `DeviceBuffer<T>`, streams) | ~3.3K | ✅ Active |
| [`cuda-host`](crates/cuda-host/README.md) | Typed module loading, launch helpers, LTOIR loader | ~1.7K | ✅ Active |
| [`cuda-async`](crates/cuda-async/README.md) | Async execution layer (`DeviceOperation`, `DeviceFuture`) | ~2.5K | ✅ Active |
| [`cuda-bindings`](crates/cuda-bindings/README.md) | Raw `bindgen` FFI to `cuda.h` | ~0.1K | ✅ Active |
| [`libnvvm-sys`](crates/libnvvm-sys/README.md) | Runtime `dlopen` bindings to libNVVM | ~0.4K | ✅ Active |
| [`nvjitlink-sys`](crates/nvjitlink-sys/README.md) | Runtime `dlopen` bindings to nvJitLink | ~0.5K | ✅ Active |

### Device Runtime Crates

| Crate | Role | LOC | Status |
|-------|------|-----|--------|
| [`cuda-device`](crates/cuda-device/README.md) | `#![no_std]` GPU intrinsics (thread, warp, TMA, atomics, tcgen05) | ~9.8K | ✅ Active |
| [`cuda-macros`](crates/cuda-macros/README.md) | Proc macros (`#[kernel]`, `#[cuda_module]`, `gpu_printf!`) | ~4.1K | ✅ Active |

### Build & Infrastructure Crates

| Crate | Role | LOC | Status |
|-------|------|-----|--------|
| [`cargo-oxide`](crates/cargo-oxide/README.md) | Cargo subcommand (`cargo oxide run`, `build`, `debug`, `pipeline`) | ~2.4K | ✅ Active |
| [`oxide-artifacts`](crates/oxide-artifacts/README.md) | Embedded device-artifact container format | ~0.9K | ✅ Active |
| [`reserved-oxide-symbols`](crates/reserved-oxide-symbols/README.md) | Internal naming contract between macros and runtime | ~0.5K | 🔒 Internal |
| [`fuzzer`](crates/fuzzer/README.md) | Differential codegen fuzzer (rustlantis adapter) | ~0.3K | 🧪 Experimental |

**Total workspace Rust:** ~59K lines (excluding examples and tests).

## Project Status

cuda-oxide is an experimental compiler that demonstrates how CUDA SIMT kernels can be written natively in pure Rust -- no DSLs, no foreign language bindings -- and made available to the broader Rust community. The project is in an early stage (alpha) and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That said, we hope you'll try it in your own work and help shape its direction by sharing feedback on your experience.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing to the project.

## Quick Start

```rust
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

// Device: generic kernel that applies any function to each element.
// F can be a closure with captures — rustc monomorphizes it to a concrete type.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[i]);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let input = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    let module = kernels::load(&ctx).unwrap();

    // Launch with a closure — factor is captured and passed to the GPU automatically
    let factor = 2.5f32;
    module
        .map::<f32, _>(
            &stream,
            LaunchConfig::for_num_elems(1024),
            move |x: f32| x * factor,
            &input,
            &mut output,
        )
        .unwrap();

    let result = output.to_host_vec(&stream).unwrap();
    assert!((result[1] - 2.5).abs() < 1e-5);
}
```

The above example defines a generic `#[kernel]` function `map` that accepts any
`Fn(T) -> T` closure. `#[cuda_module]` embeds the generated device artifact into
the host binary and generates a typed `module.map::<f32, _>(...)` launch method.
The closure `move |x| x * factor` is captured, scalarized, and passed as kernel
parameters automatically.

For composable async GPU work, `stream:` disappears, `{kernel}_async` returns a
lazy `DeviceOperation`, and execution happens when you call `.sync()` or
`.await`.

```rust
use cuda_async::device_operation::DeviceOperation;

// Assuming `module`, `input`, and `output` come from the cuda-async setup:
let factor = 2.5f32;
module
    .map_async::<f32, _>(
        LaunchConfig::for_num_elems(1024),
        move |x: f32| x * factor,
        &input,
        &mut output,
    )?
    .sync()?;
// or: .await?;
```

See the `async_mlp` example and `crates/cuda-async/README.md` for the full async setup.

```bash
# Build and run an example
cargo oxide run host_closure

# Show full compilation pipeline (Rust MIR → dialect-mir → mem2reg → LLVM dialect → LLVM IR → PTX)
cargo oxide pipeline vecadd

# Debug with cuda-gdb
cargo oxide debug vecadd --tui
```

## Setup

### Requirements

- **cargo-oxide** — cargo subcommand that drives the build pipeline (`cargo oxide run`, `build`, `debug`, etc.)
- **Rust nightly** with `rust-src` and `rustc-dev` and `llvm-tools` components (pinned in `rust-toolchain.toml`)
- **CUDA Toolkit** (12.x+)
- **Clang + libclang dev headers** (`clang-21` / `libclang-common-21-dev`) — needed by `bindgen` when building the host `cuda-bindings` crate
- **Linux** (tested on Ubuntu 24.04)

### Install

#### cargo-oxide

Inside the cuda-oxide repo, `cargo oxide` works out of the box via a workspace alias.

For use outside the repo (your own projects), install it with the pinned nightly toolchain:

```bash
cargo +nightly-2026-04-03 install --git https://github.com/SuperInstance/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend.

#### Nix (alternative)

If you have Nix with flakes enabled, `nix develop` in the repo gives you a reproducible shell with CUDA 13, LLVM 22, Clang, and the pinned Rust nightly — no manual apt installs. The shellHook auto-discovers host NVIDIA drivers on NixOS and non-NixOS systems.

```bash
nix develop                                       # full dev shell in this repo
nix run github:SuperInstance/cuda-oxide#new my-project   # bootstrap a project
```

#### Rust

```bash
# Toolchain installed automatically via rust-toolchain.toml
# Manual install if needed:
rustup toolchain install nightly-2026-04-03
rustup component add rust-src rustc-dev --toolchain nightly-2026-04-03
```

#### CUDA

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
```

#### LLVM (optional)

```bash
# Ubuntu/Debian
sudo apt install llvm-21
```

If your distro packages do not provide `llvm-21`, use LLVM's apt helper:

```bash
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

```bash
# Verify NVPTX support
llc-21 --version | grep nvptx
```

The pipeline prefers `llc` in Rust toolchain, and auto-discovers `llc-22` and `llc-21` on `PATH` (in that order).
To pin a specific binary, set `CUDA_OXIDE_LLC=/usr/bin/llc-21`.

> We emit TMA / tcgen05 / WGMMA intrinsics that `llc` from LLVM 20 and earlier can't handle.
> Simple kernels might still work with an older `llc`, but anything Hopper / Blackwell needs 21+.

#### Clang (host `cuda-bindings`)

The host `cuda-bindings` crate runs `bindgen`, which loads libclang and needs
clang's own resource-dir `stddef.h` — a bare `libclang1-*` runtime is not
enough.

```bash
sudo apt install clang-21   # or libclang-common-21-dev
```

`cargo oxide doctor` catches this up front; the symptom otherwise is a cryptic
`'stddef.h' file not found` during the host build.

#### Dev Container

The repository includes a standard devcontainer setup in `.devcontainer/` for a
reproducible CUDA, LLVM, Clang, and Rust environment. See the
[installation chapter](cuda-oxide-book/getting-started/installation.md#dev-container)
for editor and CLI usage.

### Verifying Installation

```bash
# Check that all prerequisites are in place
cargo oxide doctor

# Build and run an example end-to-end
cargo oxide run vecadd
```

`cargo oxide doctor` validates your Rust toolchain, CUDA toolkit, LLVM, and
codegen backend. If everything is configured correctly, `cargo oxide run vecadd`
compiles a Rust kernel to PTX, launches it on the GPU, and prints
`✓ SUCCESS: All 1024 elements correct!`.

## Examples

**60+ examples** in `crates/rustc-codegen-cuda/examples/`. Highlights:

| Example              | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| `vecadd`             | Vector addition -- canonical first example                               |
| `host_closure`       | Generic kernels with closures passed from host                           |
| `generic`            | Generic kernels with monomorphization (`scale<T>`)                       |
| `gemm_sol`           | GEMM SoL: 868 TFLOPS (58% cuBLAS on B200), 8 kernels across 4 phases     |
| `tcgen05`            | Blackwell tensor cores (sm_100a): TMEM, MMA, cta_group::2                |
| `atomics`            | GPU atomics: 6 types x 3 scopes x 5 orderings (20 tests)                 |
| `cluster`            | Thread Block Clusters + DSMEM ring exchange (Hopper+)                    |
| `async_mlp`          | Async MLP pipeline: GEMM → MatVec → ReLU across concurrent streams       |
| `mathdx_ffi_test`    | cuFFTDx thread-level FFT + cuBLASDx block-level GEMM                     |
| `device_ffi_test`    | Device FFI: Rust kernels calling C++ CCCL warp-level reductions via LTOIR|
| `async_vecadd`       | Async GPU execution with `cuda-async` and `DeviceOperation`              |
| `cross_crate_kernel` | Library crates defining kernels, bundled into binaries                   |

```bash
cargo oxide run vecadd
cargo oxide run gemm_sol
```

## Status

### Highlights:

- End-to-end Rust -> PTX compilation
- Unified single-source compilation (host + device in one file)
- Generic functions with monomorphization
- Closures with captures (move and non-move via HMM)
- User-defined structs, enums, pattern matching
- Full GPU intrinsic support (thread, warp, shared memory, barriers, TMA, clusters, atomics)
- Cross-crate kernels
- LTOIR generation for Blackwell+ (device-side LTO)
- Device FFI: Rust <-> C++/CCCL interop via LTOIR
- MathDx integration: cuFFTDx thread-level FFT, cuBLASDx block-level GEMM
- Tile interop (experimental): [`cutile_inter_kernel`](crates/rustc-codegen-cuda/examples/cutile_inter_kernel/README.md) chains a cutile-rs Tile kernel and a cuda-oxide SIMT PTX kernel on the same CUDA stream over shared device tensors. Intra-kernel Tile interop is work in progress and tracked in [#96](https://github.com/NVlabs/cuda-oxide/issues/96).
- Host runtime: `cuda-core` (explicit control, pinned host transfers) and `cuda-async` (composable async operations)
- GEMM SoL: 868 TFLOPS (58% cuBLAS SoL) on B200 with cta_group::2, CLC, 4-stage pipeline

## Documentation

**WIP:** 🚧 The **[cuda-oxide book](https://nvlabs.github.io/cuda-oxide/)** is the primary reference for the project. It covers SIMT kernel authoring in Rust, synchronous and asynchronous GPU programming, the compiler architecture, and more.

To build and serve the book locally, see [cuda-oxide-book/README.md](./cuda-oxide-book/README.md).

For fork-specific architecture deep-dives, see:
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — workspace composition, runtime architecture, IR dialect design, data flow
- [`PIPELINE.md`](PIPELINE.md) — step-by-step compiler pipeline walkthrough

## Contributing to the Fork

We welcome contributions that align with SuperInstance's systems-level focus:

1. **Compiler robustness** — bug fixes, additional MIR coverage, lowering correctness
2. **Runtime hardening** — `cuda-core` safety comments, `cuda-async` scheduler improvements, VMM APIs
3. **Documentation** — crate READMEs, architecture diagrams, example tutorials
4. **Systems integration** — agent runtime bindings, memory-virtualization hooks, bare-metal orchestration

Please open issues and PRs against **this fork** (`SuperInstance/cuda-oxide`). For upstream changes that should also land in NVlabs/cuda-oxide, we will coordinate back-porting.

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, commit conventions, and the PR checklist.

## Ecosystem

cuda-oxide is one of several Rust + GPU efforts under active development. Projects in this space address different parts of the problem — Vulkan/SPIR-V for graphics, implicit offload via LLVM, third-party CUDA backends, safe driver bindings — and we've been working with maintainers across the broader Rust GPU community on how to move GPU computing in Rust forward together. For where cuda-oxide fits relative to other projects, see the [Ecosystem appendix](https://nvlabs.github.io/cuda-oxide/appendix/ecosystem.html) of the book.

## License

The `cuda-bindings` crate is licensed under the NVIDIA Software License: [LICENSE-NVIDIA](LICENSE-NVIDIA). All other crates are licensed under the Apache License, Version 2.0: [LICENSE-APACHE](LICENSE-APACHE).
