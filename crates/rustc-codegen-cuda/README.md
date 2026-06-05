# rustc-codegen-cuda

A **custom rustc codegen backend** that enables single-source CUDA programming
in pure Rust. It intercepts rustc's code-generation phase, extracts functions
marked with `#[kernel]` and `#[device]`, compiles them to GPU PTX through the
cuda-oxide pipeline, and embeds the result into the host binary. Everything
else — all host code — delegates to the standard LLVM backend.

## What it replaces

Normally, rustc compiles the entire crate through `rustc_codegen_llvm`, which
produces x86_64 (or ARM64) machine code. With `rustc-codegen-cuda`, the
compilation pipeline splits:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STANDARD RUSTC PIPELINE                              │
│                                                                             │
│   Source ──▶ Parse ──▶ HIR ──▶ TypeCheck ──▶ MIR ──▶ MIR opt ──▶ LLVM ──▶ Host binary
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      CUDA-OXIDE PIPELINE (THIS BACKEND)                     │
│                                                                             │
│   Source ──▶ Parse ──▶ HIR ──▶ TypeCheck ──▶ MIR ──▶ MIR opt ──▶ ┌──────┐ │
│                                                                    │ Split │ │
│                                                                    └───┬──┘ │
│                        ┌───────────────────────────────────────────────┘    │
│                        ▼                                                    │
│   Device functions ──▶ mir-importer ──▶ dialect-mir ──▶ mem2reg ──▶ LLVM   │
│                        dialect ──▶ llvm-export ──▶ .ll ──▶ llc ──▶ PTX     │
│                                                                             │
│   Host functions ─────▶ rustc_codegen_llvm ──▶ x86_64 machine code         │
│                                                                             │
│   Final link ──▶ Host binary + embedded `.oxart` section (PTX/LTOIR/cubin) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The backend does **not** replace LLVM. It *wraps* it. Host code gets the full,
battle-tested LLVM backend. Device code gets a specialised pipeline that
understands GPU semantics — barriers, warp collectives, tensor memory,
address spaces — that LLVM's generic NVPTX backend alone cannot express from
Rust MIR.

## How it hooks into rustc

rustc loads codegen backends as dynamic libraries. The entry point is a
C-ABI symbol named `__rustc_codegen_backend`:

```text
rustc -Z codegen-backend=path/to/librustc_codegen_cuda.so
       │
       ├──▶ dlopen("librustc_codegen_cuda.so")
       │
       ├──▶ dlsym("__rustc_codegen_backend")
       │
       └──▶ __rustc_codegen_backend()
               │
               ├──▶ CudaCodegenConfig::from_env()     ← read CUDA_OXIDE_* vars
               │
               ├──▶ rustc_codegen_llvm::LlvmCodegenBackend::new()
               │
               └──▶ Return Box<CudaCodegenBackend>
```

`CudaCodegenBackend` implements rustc's [`CodegenBackend`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/traits/trait.CodegenBackend.html)
trait. Almost every method delegates straight to the wrapped LLVM backend.
The interception happens in **one method**: `codegen_crate()`.

### `codegen_crate()` — the split point

When rustc calls `codegen_crate(TyCtxt)`, the backend:

1. **Scans** the monomorphised codegen units for functions whose names contain
   the reserved `cuda_oxide_kernel_` prefix (set by the `#[kernel]` proc macro).
2. **Collects** all device-reachable functions by walking the MIR call graph
   transitively from each kernel entry point (`collector.rs`).
3. **Compiles** the collected MIR through the cuda-oxide pipeline
   (`device_codegen.rs`):
   - Enter `rustc_public` (stable MIR) context via `rustc_internal::run()`.
   - Convert `rustc_middle::ty::Instance` to stable MIR instances.
   - Call `mir_importer::run_pipeline()`:
     - Rust MIR → `dialect-mir` (alloca form)
     - `mem2reg` promotion to SSA
     - `mir-lower` → LLVM dialect
     - `llvm-export` → textual LLVM IR (`.ll`)
     - `llc -march=nvptx64` → PTX (`.ptx`)
4. **Embeds** the generated artifact into a host relocatable object with an
   `.oxart` section (`oxide-artifacts`), and appends that object to rustc's
   compiled module list so it is linked into the final host binary.
5. **Delegates** all remaining host codegen to `rustc_codegen_llvm`.

The result is a single host executable that contains both native x86_64 code
and the GPU kernels needed at runtime.

## Architecture

```text
src/
├── lib.rs              # CodegenBackend impl, __rustc_codegen_backend,
│                       # artifact embedding, panic recovery
├── collector.rs        # Device function discovery and call-graph walk
└── device_codegen.rs   # Bridge: rustc_middle ↔ stable_mir ↔ mir-importer
```

### Collector (`collector.rs`)

The collector is a worklist-driven MIR call-graph walker. Starting from kernel
entry points (functions prefixed with `cuda_oxide_kernel_246e25db_`), it
recursively discovers every function those kernels can reach.

Key design points:

- **Monomorphisation-aware**: generic kernels like `scale<T>` are skipped at
  the definition level; only concrete instantiations (`scale::<f32>`) are
  collected. The collector uses `Instance::try_resolve` with the caller's
  generic substitutions to produce fully monomorphised callees.
- **Cross-crate support**: kernels defined in library crates are discovered
  when monomorphised in the consuming crate, enabling reusable kernel
  libraries.
- **Closure collection**: when the MIR contains `FnOnce::call_once` on a
  closure type, the collector extracts the closure body directly so that the
  MIR importer can emit a direct call.
- **no_std enforcement**: the collector hard-rejects any function whose
  originating crate is `std`. This is a compile-time error, not a silent
  omission. `core` items (iterators, atomics, intrinsics) are allowed.
- **Export name generation**: kernel names are hashed from their generic type
  arguments so that `module.scale::<f32>()` on the host resolves to the exact
  PTX entry produced for `scale::<f32>` on the device.

### Device Codegen Bridge (`device_codegen.rs`)

This module solves the **two-MIR problem**: rustc's codegen backend receives
`rustc_middle` types, but the cuda-oxide pipeline was built on `rustc_public`
(stable MIR) to minimise breakage across nightly updates.

The bridge:
1. Calls `rustc_internal::run(tcx, || { ... })` to enter the stable-MIR
   translation context.
2. Converts each `CollectedFunction`'s `rustc_middle::ty::Instance` to a
   `rustc_public::mir::mono::Instance` via `rustc_internal::stable()`.
3. Extracts LLVM type strings for `#[device] extern "C"` declarations so that
   the exporter can emit correct `declare` statements.
4. Invokes `mir_importer::run_pipeline()` and maps the `CompilationResult`
   back to `DeviceCodegenResult`.

### Artifact Embedding (`lib.rs`)

After the pipeline produces PTX (or NVVM IR / LTOIR / cubin), the backend:
1. Builds an `ArtifactBundle` naming the crate, the GPU target (`sm_80`,
   `sm_90a`, `sm_100a`, …), the payload, and every kernel/device entry.
2. Serialises the bundle via `oxide-artifacts::build_artifact_blob()`.
3. Wraps it in an ELF object via `oxide-artifacts::build_host_object_for_target()`.
4. Appends the object to `CompiledModules.modules` so rustc's linker includes
   it.

At runtime, `cuda-host` reads the `.oxart` section from the loaded executable
and loads the payload through the CUDA driver.

### Panic Recovery

Because the cuda-oxide pipeline (pliron IR, lowering, export) can panic on
invalid input or internal invariants, the backend wraps
`device_codegen::generate_device_code()` in `catch_unwind`. If a panic
escapes, the backend:

1. Temporarily replaces rustc's panic hook to capture a backtrace (the hook
   fires before `catch_unwind` sees the unwind).
2. Emits a fatal rustc diagnostic pointing users to the cuda-oxide issue
tracker rather than rustc's.

This prevents a backend bug from being misreported as an ICE in rustc itself.

## Constraints and Design Decisions

### Nightly Rust only

The backend uses `#![feature(rustc_private)]` to access rustc internals
(`rustc_middle`, `rustc_codegen_ssa`, `rustc_codegen_llvm`, `rustc_public`).
It pins to a specific nightly (`nightly-2026-04-03` at the time of writing)
via `rust-toolchain.toml`.

### Required compiler flags

`cargo oxide` sets these automatically. For manual invocations they are
required:

| Flag | Purpose |
|------|---------|
| `-C opt-level=3` | Enables MIR inlining, const-prop, and GVN for device code |
| `-C debug-assertions=off` | Strips `debug_assert!` paths that pull in `core::fmt` |
| `-Z mir-enable-passes=-JumpThreading` | **Critical**: prevents barrier duplication across branches |

**Why disable JumpThreading?** JumpThreading duplicates basic blocks to
eliminate jumps. On GPU code this can copy a `__syncthreads()` into two
branches. Threads that take different paths then execute *different* barrier
instances → **deadlock**. The backend treats all unwind edges as unreachable,
so `panic=abort` and custom sysroots are **not** required.

### Device code is `no_std`

Functions reachable from a `#[kernel]` may only call into `core`,
`cuda_device`, or the local crate. The collector enforces this with a hard
error. `core` items shown as `std::*` in MIR dumps are cosmetic re-exports —
the actual `DefId` lives in `core` and is allowed.

### Argument scalarisation

Aggregates (slices, structs, closures) are flattened to scalar LLVM
parameters at the host/device boundary and reconstructed inside the kernel.
This matches the CUDA launch ABI and is transparent to the user.

### Struct layout compatibility

Device-side structs use explicit padding derived from rustc's layout queries,
so `#[repr(C)]` is not required for host/device ABI compatibility.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `CUDA_OXIDE_VERBOSE` | Print compilation progress and collected functions |
| `CUDA_OXIDE_DUMP_MIR` | Dump the `dialect-mir` module before lowering |
| `CUDA_OXIDE_DUMP_LLVM` | Dump the LLVM dialect module |
| `CUDA_OXIDE_SHOW_RUSTC_MIR` | Dump raw rustc MIR before translation |
| `CUDA_OXIDE_PTX_DIR` | Output directory for `.ptx` and `.ll` files |
| `CUDA_OXIDE_TARGET` | Override GPU architecture (`sm_80`, `sm_90a`, `sm_100a`, …) |
| `CUDA_OXIDE_LLC` | Path to a specific `llc` binary |
| `CUDA_OXIDE_EMIT_NVVM_IR` | Skip `llc`; emit NVVM IR for libNVVM / LTOIR |

## Usage

```bash
# Preferred: use the cargo-oxide wrapper
cargo oxide run vecadd

# See the full compilation pipeline
cargo oxide pipeline vecadd

# Manual invocation
RUSTFLAGS="-Z codegen-backend=path/to/librustc_codegen_cuda.so \
           -C opt-level=3 \
           -C debug-assertions=off \
           -Z mir-enable-passes=-JumpThreading" \
    cargo run --release
```

## Examples

The `examples/` directory contains **60+** standalone kernel crates. Highlights:

| Example | What it covers |
|---------|----------------|
| `vecadd` | Basic vector addition — the "hello world" kernel |
| `generic` | Generic kernels with monomorphisation (`scale<T>`) |
| `host_closure` | Closures with captures passed from host |
| `cross_crate_kernel` | Kernels defined in a library crate |
| `atomics` | GPU atomics: 6 types × 3 scopes × 5 orderings |
| `tma_copy` / `tma_multicast` | Tensor Memory Accelerator (Hopper+) |
| `wgmma` | Warpgroup MMA (Hopper tensor cores) |
| `tcgen05` / `tcgen05_matmul` | 5th-gen tensor cores + TMEM (Blackwell) |
| `gemm_sol` | GEMM at 868 TFLOPS (58% cuBLAS SoL on B200) |
| `cluster` / `clc` | Thread-block clusters and Cluster Launch Control |
| `device_ffi_test` | Rust kernels calling C++/CCCL via LTOIR |
| `mathdx_ffi_test` | cuFFTDx thread-level FFT + cuBLASDx block GEMM |

Run any example with:

```bash
cargo oxide run <example_name>
```

## License

Licensed under the Apache License, Version 2.0.
