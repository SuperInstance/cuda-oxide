# nvjitlink-sys

Runtime (`dlopen`) bindings to NVIDIA's **nvJitLink**.

nvJitLink is a linker for GPU code. It consumes one or more input modules —
LTOIR, PTX, cubin, fatbin, object files, or libraries — and links them into a
single output cubin (or PTX). It performs device-side link-time optimization
(LTO), dead-code elimination across modules, and SM-specific code generation.

In the cuda-oxide pipeline, nvJitLink is the **final stage** when using the
libNVVM path: kernel NVVM IR is compiled to LTOIR by libNVVM, then fed into
nvJitLink along with `libdevice.10.bc` and any external LTOIR objects, producing
a cubin that the CUDA driver can load directly.

## Where nvJitLink fits in the pipeline

```text
Rust MIR ──▶ dialect-mir ──▶ LLVM dialect ──▶ LLVM IR
                                                    │
                    ┌───────────────────────────────┘
                    │  (NVVM IR mode — no llc)
                    ▼
              libNVVM (libnvvm-sys)
                    │
                    └──▶ LTOIR bytes
                    │
                    ▼
              nvJitLink (THIS CRATE)
                    │
                    ├──▶ add(Ltoir, kernel.ltoir)
                    ├──▶ add(Ltoir, libdevice.10.bc)
                    ├──▶ add(Ltoir, extern_math.ltoir)   ← device FFI
                    └──▶ finish()
                    │
                    ▼
                 cubin bytes
                    │
                    ▼
              cuModuleLoadDataEx ──▶ kernel handle
```

The `-lto` option must be passed to `Linker::new` so that nvJitLink knows to
enable link-time optimization; without it, LTOIR inputs are rejected.

## What this crate provides

- **`LibNvJitLink`** — RAII wrapper around the loaded library + resolved
  function pointers. Owns the `dlopen` handle.
- **`Linker`** — RAII wrapper around an `nvJitLinkHandle`. Create, add inputs,
  and finish to produce a cubin.
- **`InputType`** — supported input formats (`Ltoir`, `Ptx`, `Cubin`,
  `Fatbin`, `Object`, `Library`, `Index`, `Any`).
- **`NvJitLinkError`** — typed errors with the nvJitLink error log attached.

## Build requirements

**None.** Like `libnvvm-sys`, this crate loads the library at runtime. The CUDA
Toolkit only needs to be present when the program runs.

## Library discovery

`LibNvJitLink::load()` tries (in order):

1. `LIBNVJITLINK_PATH` environment variable, if set.
2. The system loader (`libnvJitLink.so.13`, `libnvJitLink.so.12`,
   `libnvJitLink.so`).
3. `<root>/lib64/libnvJitLink.so` for `<root>` in `CUDA_HOME`, `CUDA_PATH`,
   `/usr/local/cuda`, `/opt/cuda`.

nvJitLink ships with the standard CUDA Toolkit at `<cuda>/lib64/`.

## Symbol naming

`nvJitLink.h` `#define`s every public function to a versioned mangled name
(e.g. `nvJitLinkCreate -> __nvJitLinkCreate_13_0`), but the library also
exports the **unversioned name** with default ELF symbol versioning.
`dlsym(handle, "nvJitLinkCreate")` resolves to the right function on every
CUDA Toolkit version, so this binding does not need to probe per-CUDA-version
symbol suffixes.

An optional `nvJitLinkVersion` symbol (added in CTK 12.4) is resolved with
`dlsym` as well; if absent, `LibNvJitLink::version()` returns `None`.

## Usage

This crate is low-level. Most users want the higher-level
`cuda_host::ltoir::load_kernel_module` helper, which combines libNVVM +
libdevice + nvJitLink behind one call. Use this crate directly only if you
need explicit control over the link.

```rust
use nvjitlink_sys::{LibNvJitLink, Linker, InputType};

let nvj = LibNvJitLink::load()?;
let mut linker = Linker::new(&nvj, &["-arch=sm_120", "-lto"])?;
linker.add(InputType::Ltoir, &ltoir_bytes, "kernel.ltoir")?;
let cubin = linker.finish()?;
```

When `CUDA_OXIDE_VERBOSE` is set, the nvJitLink info log (timings, chosen SM,
etc.) is printed to `stderr` at the end of `finish()`.

## Companion crate

[`libnvvm-sys`](../libnvvm-sys/) — same pattern, for libNVVM. Together they
cover the NVVM IR → LTOIR → cubin pipeline.
