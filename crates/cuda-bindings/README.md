# cuda-bindings

Raw FFI bindings to the CUDA Driver API (`cuda.h`), generated at build time by
[bindgen](https://crates.io/crates/bindgen).

This is the lowest layer of the cuda-oxide host stack. Every higher-level crate
— `cuda-core`, `cuda-async`, `cuda-host` — builds on top of these bindings.
You should almost never use this crate directly; prefer the safe wrappers
upstream. It exists because someone has to hold the `unsafe` so that everyone
else doesn't have to.

## What this crate provides

- **All types, constants, and function declarations from `cuda.h`** — e.g.
  `cuInit`, `cuLaunchKernel`, `CUdeviceptr`, `CUstream`, `CUresult`,
  `CUDA_SUCCESS`, …
- **`cuda_toolkit_dir()`** — a runtime helper that resolves the CUDA toolkit
  root using the same logic as the build script (`CUDA_TOOLKIT_PATH` →
  `/usr/local/cuda`).
- **`cu_event_elapsed_time`** — a thin compatibility shim that calls
  `cuEventElapsedTime_v2` on CUDA 12.8+ and falls back to
  `cuEventElapsedTime` on older toolkits.

## Build requirements

- **CUDA Toolkit** installed with `include/cuda.h` present.
- **Clang** available for bindgen (the `libclang` runtime alone is not enough;
  you need Clang's resource-dir headers such as `stddef.h`).

The build script:
1. Locates the toolkit via `CUDA_TOOLKIT_PATH` (or `CUDA_HOME`, then
   `/usr/local/cuda`).
2. Emits `cargo:rustc-link-search` for discovered library directories
   (`lib64/`, `targets/x86_64-linux/lib/`, and their `stubs/` subdirs).
3. Links `libcuda` (`dylib=cuda`) — the driver stub that forwards to the
   real driver installed by the NVIDIA kernel module.
4. Runs bindgen on `wrapper.h` with `-I{toolkit}/include`.

## Why `bindgen`?

CUDA's driver API is a large, versioned C API. Maintaining hand-written FFI
bindings would be error-prone and would drift with every CUDA Toolkit release.
`bindgen` reads the headers directly, so upgrading the toolkit automatically
upgrades the Rust signatures (subject to the opaque-type workarounds in
`build.rs` for structs that libclang cannot translate).

## Environment variables

| Variable            | Purpose                                    | Default            |
|---------------------|--------------------------------------------|--------------------|
| `CUDA_TOOLKIT_PATH` | Root of the CUDA Toolkit installation      | `/usr/local/cuda`  |

## License

This crate is licensed under the NVIDIA Software License. See
[`LICENSE-NVIDIA`](../../LICENSE-NVIDIA).
