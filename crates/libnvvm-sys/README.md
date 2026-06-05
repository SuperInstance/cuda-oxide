# libnvvm-sys

Runtime (`dlopen`) bindings to NVIDIA's **libNVVM**.

libNVVM is the front-end of NVIDIA's PTX-targeting compiler. It accepts
**NVVM IR** — an LLVM-IR dialect with NVIDIA-specific intrinsics — and
produces either textual PTX or binary **LTOIR** (Link-Time Optimization IR).
In the cuda-oxide pipeline, libNVVM is used when a kernel calls `__nv_*`
libdevice math functions (e.g. `__nv_sinf`) that LLVM's NVPTX backend cannot
resolve on its own.

## Where libNVVM fits in the pipeline

```text
Rust MIR ──▶ dialect-mir ──▶ LLVM dialect ──▶ LLVM IR (.ll)
                                                    │
                    ┌───────────────────────────────┘
                    │  (no llc — skip PTX generation)
                    ▼
              libNVVM (this crate)
                    │
                    ├──▶ add_module(kernel.ll)
                    ├──▶ add_module(libdevice.10.bc)
                    └──▶ compile("-arch=compute_120", "-gen-lto")
                    │
                    ▼
                 LTOIR bytes
                    │
                    ▼
              nvJitLink (nvjitlink-sys)
                    │
                    └──▶ linked cubin ──▶ cuModuleLoad
```

When the module contains `__nv_*` externals, the pipeline switches from the
default `llc` path to the **NVVM IR → LTOIR → cubin** path. `cuda-host`'s
`ltoir::load_kernel_module` helper automates this entire flow.

## What this crate provides

- **`LibNvvm`** — RAII wrapper around the loaded library + resolved function
  pointers. Owns the `dlopen` handle; dropping it unloads the library.
- **`Program`** — RAII wrapper around an `nvvmProgram` handle. Add modules,
  compile, and retrieve the result.
- **`NvvmError`** — typed errors that capture the libNVVM program log so that
  compilation failures are actionable.

## Build requirements

**None.** The library is loaded at runtime via `libloading`, so the CUDA
Toolkit only needs to be present when the program runs, not when it compiles.
This keeps the build graph small and avoids hard system dependencies.

## Library discovery

`LibNvvm::load()` tries (in order):

1. `LIBNVVM_PATH` environment variable, if set.
2. The system loader (`libnvvm.so.4`, `libnvvm.so.3`, `libnvvm.so`).
3. `<root>/nvvm/lib64/libnvvm.so` for `<root>` in `CUDA_HOME`, `CUDA_PATH`,
   `/usr/local/cuda`, `/opt/cuda`.

libNVVM ships with the standard CUDA Toolkit at `<cuda>/nvvm/lib64/`.
No separate download is required.

## Symbol naming

libNVVM uses plain unversioned symbol names (`nvvmCreateProgram`,
`nvvmCompileProgram`, …), so a single `dlsym` lookup per function is
sufficient across CUDA versions.

## Usage

This crate is low-level. Most users want the higher-level
`cuda_host::ltoir::load_kernel_module` helper, which combines libNVVM +
`libdevice.10.bc` + nvJitLink behind one call. Use this crate directly only
if you need explicit control over the libNVVM compile.

```rust
use libnvvm_sys::{LibNvvm, Program};

let nvvm = LibNvvm::load()?;
let mut program = Program::new(&nvvm)?;
program.add_module(&libdevice_bytes, "libdevice.10.bc")?;
program.add_module(&kernel_ll_bytes, "kernel.ll")?;
let ltoir = program.compile(&["-arch=compute_120", "-gen-lto"])?;
```

## Companion crate

[`nvjitlink-sys`](../nvjitlink-sys/) — same pattern, for nvJitLink. Together
they cover the NVVM IR → LTOIR → cubin pipeline.
