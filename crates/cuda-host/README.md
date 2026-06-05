# cuda-host

Host-side runtime for cuda-oxide. This crate sits between the CUDA driver
(`cuda-core`) and the proc macros (`cuda-macros`). It provides typed module
loading, kernel launch infrastructure, device memory helpers, and the LTOIR
build pipeline for kernels that use CUDA libdevice math.

## Relationship to cuda-core

| Crate | Responsibility |
|-------|---------------|
| `cuda-core` | Raw CUDA driver API: `CudaContext`, `CudaStream`, `CudaModule`, `DeviceBuffer`, `launch_kernel_on_stream`, embedded artifact sections |
| `cuda-host` | Typed convenience layer: `#[cuda_module]` loader, `CudaKernel` / `GenericCudaKernel` traits, LTOIR-to-cubin pipeline, tensor-core tiling utilities |
| `cuda-async` | Futures-based scheduling on top of `cuda-host` + `cuda-core` |

`cuda-host` re-exports the `#[cuda_module]` and `cuda_launch!` macros from
`cuda-macros`, and adds the traits and helpers those macros expand into.

## Memory management

### `DeviceBuffer<T>` (from `cuda-core`)

The primary typed device allocation. `cuda-host` does not reimplement allocation;
it relies on `cuda-core::DeviceBuffer` and adds kernel-argument marshalling
helpers around it.

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

let ctx = CudaContext::new(0)?;
let stream = ctx.default_stream();

// Allocate and upload
let a_dev = DeviceBuffer::from_host(&stream, &vec![1.0f32; 1024])?;
let mut b_dev = DeviceBuffer::<f32>::zeroed(&stream, 1024)?;

// Download
let b_host: Vec<f32> = b_dev.to_host_vec(&stream)?;
```

### Kernel argument marshalling

`cuda-host` provides the boundary between Rust types and the `Vec<*mut c_void>`
that `cuLaunchKernel` expects:

| Rust type | Kernel parameter | Marshalling |
|-----------|-----------------|-------------|
| `T: Copy` (scalar, struct, closure) | `T` | `&mut value` as `*mut c_void` |
| `&DeviceBuffer<T>` | `&[T]` | `(CUdeviceptr, u64)` — two args |
| `&mut DeviceBuffer<T>` | `&mut [T]` or `DisjointSlice<T>` | `(CUdeviceptr, u64)` — two args |

ZST scalars (e.g., zero-capture closures, unit structs) are automatically
skipped on both host and device so the parameter indices stay aligned.

## Kernel launching mechanics

### The `#[cuda_module]` generated API

Place `#[cuda_module]` on an inline module containing `#[kernel]` functions.
The macro generates:

```rust
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx.get()] + b[idx.get()];
        }
    }
}
```

Generated items:

| Item | Purpose |
|------|---------|
| `LoadedModule` | Typed handle around the embedded CUDA module and cached kernel functions |
| `load(&Arc<CudaContext>)` | Load the current crate's embedded artifact bundle |
| `load_named(&Arc<CudaContext>, name)` | Load a specific embedded bundle by name |
| `from_module(Arc<CudaModule>)` | Wrap an already-loaded CUDA module |
| `LoadedModule::{kernel}` | One typed launch method per `#[kernel]` function |
| `load_async(device_id)` | With feature `async`, load from a `cuda-async` device context |
| `LoadedModule::{kernel}_async` | With feature `async`, build a lazy `AsyncKernelLaunch` |
| `LoadedModule::{kernel}_async_owned` | With feature `async`, owned async launch that returns buffers |

### Synchronous launch

```rust
let ctx = CudaContext::new(0)?;
let stream = ctx.default_stream();
let module = kernels::load(&ctx)?;

module.vecadd(
    &stream,
    LaunchConfig::for_num_elems(1024),
    &a_dev,
    &b_dev,
    &mut c_dev,
)?;
```

The launch method:
1. Looks up the cached `CudaFunction` (or loads it from the module).
2. Marshals arguments into a `Vec<*mut c_void>`.
3. Calls `cuda_core::launch_kernel_on_stream` (or `launch_kernel_ex_on_stream` for cluster launches).

### Generic kernels and PTX name resolution

Non-generic kernels have a fixed PTX name (`vecadd`). Generic kernels are named
`<base>_TID_<hex32>`, where `<hex32>` is a 32-char lowercase hex hash of the
tuple of generic type parameters. Both the backend (inside `rustc_codegen_cuda`)
and the host (`cuda_host::type_id_u128::<(T0, T1, ...)>()`)
compute the same hash via `tcx.type_id_hash`, so the strings match byte-for-byte.

The `GenericCudaKernel` trait provides `ptx_name() -> &'static str` for runtime
lookup, and a volatile-pointer monomorphization trick ensures the kernel appears
in the codegen unit even without a host-side call.

## Stream management

`cuda-host` does not own streams directly — that lives in `cuda-core` and
`cuda-async`. However, all synchronous launch methods take a `&CudaStream`
argument, making stream usage explicit:

```rust
let stream = ctx.new_stream()?;  // from cuda-core
module.vecadd(&stream, config, &a, &b, &mut c)?;
stream.synchronize()?;
```

When the `async` feature is enabled, `cuda-host` generates async launch methods
that defer stream selection to the `cuda-async` scheduling policy:

```rust
let module = kernels::load_async(0)?;
module
    .vecadd_async(LaunchConfig::for_num_elems(1024), &a_dev, &b_dev, &mut c_dev)?
    .sync()?;
```

## LTOIR pipeline (libNVVM + nvJitLink)

When a kernel uses Rust float math intrinsics (`sin`, `cos`, `exp`, `pow`, ...),
cuda-oxide auto-detects them, emits NVVM IR (`.ll`) instead of PTX, and skips
`llc`. At runtime `cuda-host::ltoir` builds the cubin on demand:

1. **libNVVM** compiles the NVVM IR + `libdevice.10.bc` to LTOIR.
2. **nvJitLink** links the LTOIR with `-arch=sm_XX -lto` to produce a cubin.
3. The cubin is loaded via `cuModuleLoad`.

```rust
use cuda_host::ltoir;

// Loads my_kernel.cubin, or builds it from my_kernel.ll automatically.
let module = ltoir::load_kernel_module(&ctx, "my_kernel")?;
```

Discovery:
- **libNVVM**: `LIBNVVM_PATH` → system loader → `<CUDA>/nvvm/lib64/libnvvm.so`
- **nvJitLink**: `<CUDA>/lib64/libnvJitLink.so`
- **libdevice**: `CUDA_OXIDE_LIBDEVICE` → `<CUDA>/nvvm/libdevice/libdevice.10.bc`
- **Arch**: `CUDA_OXIDE_TARGET` env var, defaulting to `sm_120`

## Embedded module loading

Device artifacts (PTX, cubin, NVVM IR, LTOIR) are embedded into the host binary
at compile time by the codegen backend. `cuda-host::embedded` reads the artifact
section and picks the best payload:

1. Cubin — load directly.
2. PTX — JIT-compile via the driver.
3. NVVM IR — build cubin via the LTOIR pipeline.
4. LTOIR — link via nvJitLink.

```rust
use cuda_host::embedded;

let module = embedded::load_embedded_module(&ctx, env!("CARGO_PKG_NAME"))?;
```

## Tensor-core tiling (tcgen05)

Host-side layout transformations for Blackwell tensor cores. tcgen05 requires
specific 8×8 tile arrangements:

| Function | Description |
|----------|-------------|
| `to_k_major_f16` | Row-major → tcgen05 K-major (matrix A) |
| `to_mn_major_f16` | Row-major → tcgen05 MN-major (matrix B) |
| `k_major_index` | Linear index in K-major layout |
| `mn_major_index` | Linear index in MN-major layout |
| `print_layout_indices` | Debug print as 2D table |
| `TILE_SIZE` | Constant `8` |

## Lower-level APIs

`CudaKernel` and `GenericCudaKernel` remain the marker traits generated by
`#[kernel]`. `cuda_launch!` is available as a lower-level migration path, but
new code should prefer `#[cuda_module]` typed methods.

`cuda_launch_async!` is also lower-level. It can describe lazy work from raw
device pointers, so callers must ensure pointed-to allocations outlive the
operation. Generated borrowed async methods encode that requirement as Rust
borrows; generated owned async methods move buffers into the operation for
spawned tasks.

## Further reading

- [cuda-device](../cuda-device/) — device-side intrinsics (`thread`, `DisjointSlice`, etc.)
- [cuda-macros](../cuda-macros/) — proc-macro implementations
- [cuda-core](../cuda-core/) — CUDA driver API, `DeviceBuffer`, `LaunchConfig`
- [cuda-async](../cuda-async/) — async scheduling and stream pools
