# cuda-bindings

Raw FFI bindings to the NVIDIA CUDA Driver API (`cuda.h`), generated at build time by [bindgen](https://crates.io/crates/bindgen). This crate is the foundation of the cuda-oxide stack: `cuda-core` builds safe RAII wrappers on top of these raw declarations, and most users should prefer `cuda-core` for day-to-day host code.

If you are writing a host application, start with [`cuda-core`](../cuda-core/). If you are writing kernels in Rust, see [`cuda-device`](../cuda-device/).

## Table of Contents

- [What This Crate Provides](#what-this-crate-provides)
- [FFI Binding Strategy](#ffi-binding-strategy)
- [Build Requirements](#build-requirements)
- [Toolkit Discovery](#toolkit-discovery)
- [What CUDA Driver API Is Exposed](#what-cuda-driver-api-is-exposed)
- [Using the Raw Bindings](#using-the-raw-bindings)
- [Version Compatibility](#version-compatibility)
- [Generated Type Conventions](#generated-type-conventions)
- [Safety](#safety)
- [Related Crates](#related-crates)

---

## What This Crate Provides

- **Every type, constant, and function declaration** from the CUDA driver API header (`cuda.h`) — context management, streams, events, memory allocation, module loading, kernel launch, peer access, virtual memory management, tensor map descriptors, and more.
- **`cuda_toolkit_dir()`**: A runtime helper that resolves the CUDA Toolkit root directory using the same search order as the build script.
- **`cu_event_elapsed_time()`**: A thin compatibility shim that calls `cuEventElapsedTime_v2` on CUDA 12.8+ and falls back to `cuEventElapsedTime` on older toolkits.
- **Automatic linking** against `libcuda.so` (the driver stub) with platform-aware library search paths.

---

## FFI Binding Strategy

The crate uses a **build-time bindgen** approach rather than hand-written `extern "C"` blocks. This ensures the Rust declarations stay in lockstep with the CUDA Toolkit installed on the build machine.

```text
wrapper.h  ──►  bindgen + clang  ──►  OUT_DIR/bindings.rs  ──►  include!
   │                  │
   │                  └── -I$CUDA_TOOLKIT_PATH/include
   │
   └── #include <cuda.h>
```

**Why bindgen?**

| Approach | Maintenance | Accuracy | Version adaptability |
|----------|-------------|----------|----------------------|
| Hand-written `extern "C"` | High | Risk of drift | Requires manual updates per CUDA release |
| **bindgen from `cuda.h`** | Low | Exact | Automatically adapts to the host's installed toolkit |

The generated bindings live in `OUT_DIR/bindings.rs` and are pulled into the crate via `include!`. They carry CUDA's original C doxygen comments verbatim, which is why rustdoc lints for broken intra-doc links and bare URLs are explicitly allowed.

**Opaque types for forward compatibility:**

CUDA 13.2+ adds fields to `CUlaunchAttributeValue_union` that libclang cannot translate. The build script marks both `CUlaunchAttribute_st` and `CUlaunchAttributeValue_union` as opaque, producing correctly-sized byte blobs across all supported CUDA versions. Higher-level code in `cuda-core` constructs these structs via raw pointer writes when needed (e.g. for cluster launches and cooperative launches).

---

## Build Requirements

- **CUDA Toolkit** installed with headers in `<toolkit>/include/cuda.h`.
- **Clang** available for bindgen (typically bundled with LLVM).
- A CUDA-capable driver installed on the system.

The build script will fail with a descriptive error if `cuda.h` cannot be found.

---

## Toolkit Discovery

The build script and the runtime `cuda_toolkit_dir()` helper use the following resolution order:

| Source | Variable / Path |
|--------|-----------------|
| 1. Environment variable | `CUDA_TOOLKIT_PATH` |
| 2. Fallback environment variable | `CUDA_HOME` (build script only) |
| 3. Default | `/usr/local/cuda` |

Changing `CUDA_TOOLKIT_PATH` or `wrapper.h` triggers an automatic rebuild.

**Library search paths:**

The build script adds the following directories to the linker search path (when they exist):
- `{toolkit}/lib64`
- `{toolkit}/lib64/stubs`
- `{toolkit}/targets/x86_64-linux/lib`
- `{toolkit}/targets/x86_64-linux/lib/stubs`

It then links `libcuda` dynamically:
```rust
println!("cargo:rustc-link-lib=dylib=cuda");
```

---

## What CUDA Driver API Is Exposed

The generated bindings cover the complete CUDA Driver API surface. Major categories include:

### Initialization & Device Management
- `cuInit`, `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceGetUuid`, `cuDeviceGetMemPool`, `cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxRelease`, `cuDevicePrimaryCtxReset`, `cuDevicePrimaryCtxSetFlags`, `cuDeviceGetTexture1DLinearMaxWidth`

### Context Management
- `cuCtxCreate`, `cuCtxDestroy`, `cuCtxPushCurrent`, `cuCtxPopCurrent`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxSynchronize`, `cuCtxGetStreamPriorityRange`, `cuCtxSetLimit`, `cuCtxGetLimit`, `cuCtxGetCacheConfig`, `cuCtxSetCacheConfig`, `cuCtxGetSharedMemConfig`, `cuCtxSetSharedMemConfig`, `cuCtxGetApiVersion`, `cuCtxGetFlags`, `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess`

### Memory Management
- `cuMemAlloc`, `cuMemAlloc_v2`, `cuMemAllocPitch`, `cuMemFree`, `cuMemFree_v2`, `cuMemGetInfo`, `cuMemGetInfo_v2`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostGetFlags`, `cuMemHostRegister`, `cuMemHostUnregister`, `cuMemcpy`, `cuMemcpyAsync`, `cuMemcpyHtoD`, `cuMemcpyHtoD_v2`, `cuMemcpyHtoDAsync`, `cuMemcpyHtoDAsync_v2`, `cuMemcpyDtoH`, `cuMemcpyDtoH_v2`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoHAsync_v2`, `cuMemcpyDtoD`, `cuMemcpyDtoD_v2`, `cuMemcpyDtoDAsync`, `cuMemcpyDtoDAsync_v2`, `cuMemsetD8`, `cuMemsetD8Async`, `cuMemsetD16`, `cuMemsetD16Async`, `cuMemsetD32`, `cuMemsetD32Async`

### Stream-Ordered Memory (CUDA 11.2+)
- `cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemPoolCreate`, `cuMemPoolDestroy`, `cuMemPoolTrimTo`, `cuMemPoolSetAttribute`, `cuMemPoolGetAttribute`, `cuMemPoolSetAccess`, `cuMemPoolGetAccess`, `cuMemPoolExportPointer`, `cuMemPoolImportPointer`, `cuMemPoolExportToShareableHandle`, `cuMemPoolImportFromShareableHandle`

### Virtual Memory Management (sm_70+)
- `cuMemCreate`, `cuMemRelease`, `cuMemAddressReserve`, `cuMemAddressFree`, `cuMemMap`, `cuMemUnmap`, `cuMemSetAccess`, `cuMemGetAccess`, `cuMemGetAllocationGranularity`, `cuMemGetAllocationPropertiesFromHandle`, `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`

### Streams
- `cuStreamCreate`, `cuStreamDestroy`, `cuStreamDestroy_v2`, `cuStreamSynchronize`, `cuStreamQuery`, `cuStreamWaitEvent`, `cuStreamAddCallback`, `cuStreamAttachMemAsync`, `cuStreamBeginCapture`, `cuStreamEndCapture`, `cuStreamIsCapturing`, `cuStreamGetCaptureInfo`, `cuStreamGetCaptureInfo_v2`, `cuStreamUpdateCaptureDependencies`, `cuStreamGetId`

### Events
- `cuEventCreate`, `cuEventDestroy`, `cuEventDestroy_v2`, `cuEventRecord`, `cuEventSynchronize`, `cuEventQuery`, `cuEventElapsedTime`, `cuEventElapsedTime_v2`

### Module & Function Management
- `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleLoadFatBinary`, `cuModuleUnload`, `cuModuleGetFunction`, `cuModuleGetGlobal`, `cuModuleGetGlobal_v2`, `cuModuleGetTexRef`, `cuModuleGetSurfRef`

### Kernel Launch
- `cuLaunchKernel`, `cuLaunchKernelEx`, `cuLaunchCooperativeKernel`, `cuLaunchHostFunc`

### Peer Access
- `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess`

### Texture & Surface Reference APIs (legacy)
- `cuTexRefCreate`, `cuTexRefDestroy`, `cuTexRefSetArray`, `cuTexRefSetAddress`, `cuTexRefSetAddress2D`, `cuTexRefSetAddressMode`, `cuTexRefSetFilterMode`, `cuTexRefSetFlags`, `cuSurfRefSetArray`

### Tensor Memory Accelerator (sm_90+)
- `cuTensorMapEncodeTiled`, `cuTensorMapEncodeIm2col`, `cuTensorMapReplaceAddress`

### Graphs (CUDA 10+)
- `cuGraphCreate`, `cuGraphDestroy`, `cuGraphAddKernelNode`, `cuGraphAddMemcpyNode`, `cuGraphAddMemsetNode`, `cuGraphAddHostNode`, `cuGraphAddChildGraphNode`, `cuGraphAddEmptyNode`, `cuGraphAddEventRecordNode`, `cuGraphAddEventWaitNode`, `cuGraphAddExternalSemaphoresSignalNode`, `cuGraphAddExternalSemaphoresWaitNode`, `cuGraphAddMemAllocNode`, `cuGraphAddMemFreeNode`, `cuGraphClone`, `cuGraphInstantiate`, `cuGraphInstantiateWithFlags`, `cuGraphLaunch`, `cuGraphUpload`, `cuGraphAddDependencies`, `cuGraphRemoveDependencies`, `cuGraphGetNodes`, `cuGraphGetRootNodes`, `cuGraphGetEdges`, `cuGraphNodeGetType`, `cuGraphNodeGetDependencies`, `cuGraphNodeGetDependentNodes`, `cuGraphExecDestroy`, `cuGraphExecKernelNodeSetParams`, `cuGraphExecMemcpyNodeSetParams`, `cuGraphExecMemsetNodeSetParams`, `cuGraphExecHostNodeSetParams`, `cuGraphExecChildGraphNodeSetParams`, `cuGraphExecEventRecordNodeSetEvent`, `cuGraphExecEventWaitNodeSetEvent`, `cuGraphExecExternalSemaphoresSignalNodeSetParams`, `cuGraphExecExternalSemaphoresWaitNodeSetParams`, `cuGraphExecUpdate`, `cuGraphExecGetFlags`

### External Resource Interop
- `cuImportExternalMemory`, `cuDestroyExternalMemory`, `cuExternalMemoryGetMappedBuffer`, `cuExternalMemoryGetMappedMipmappedArray`, `cuImportExternalSemaphore`, `cuDestroyExternalSemaphore`, `cuSignalExternalSemaphoresAsync`, `cuWaitExternalSemaphoresAsync`

### Occupancy & Launch Configuration
- `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`, `cuOccupancyMaxPotentialBlockSize`, `cuOccupancyMaxPotentialBlockSizeWithFlags`, `cuOccupancyAvailableDynamicSMemPerBlock`

---

## Using the Raw Bindings

While `cuda-core` provides safe wrappers for the common path, you can drop down to the raw bindings when you need an API that is not yet wrapped:

```rust
use cuda_bindings::{cuInit, cuDeviceGet, CUdevice, CUresult, cudaError_enum_CUDA_SUCCESS};

fn main() {
    unsafe {
        // Initialize the driver
        let result: CUresult = cuInit(0);
        assert_eq!(result, cudaError_enum_CUDA_SUCCESS);

        // Get device 0
        let mut device: CUdevice = std::mem::zeroed();
        let result = cuDeviceGet(&mut device, 0);
        assert_eq!(result, cudaError_enum_CUDA_SUCCESS);
    }
}
```

All functions are `unsafe` because they carry the standard CUDA API preconditions: valid handles, correct context binding, proper stream ordering, and so on. When possible, prefer `cuda-core` which encodes many of these invariants in the type system (RAII, context binding, error propagation).

### Re-export Path

`cuda-core` re-exports the raw bindings as `cuda_core::sys`:

```rust
use cuda_core::sys::{cuLaunchKernel, CUfunction, CUstream};
```

This is the idiomatic way to access raw APIs from host code that already uses `cuda-core`.

---

## Version Compatibility

The crate adapts to the CUDA Toolkit version present at build time:

| CUDA Version | Notable Adaptation |
|--------------|-------------------|
| 12.4 – 12.7 | `cuEventElapsedTime` (no `_v2` suffix) |
| 12.8+ | `cuEventElapsedTime_v2` — the compatibility shim selects the right symbol at build time via `cfg(cuda_has_cuEventElapsedTime_v2)` |
| 13.0+ | `CUlaunchAttributeValue_union` gains additional fields; bindgen marks it opaque to avoid translation failures |
| All supported | `CUlaunchAttribute_st` is opaque for the same forward-compatibility reason |

Because bindings are generated from the local `cuda.h`, using a newer CUDA Toolkit on the build machine automatically exposes newer API entries. Using an older Toolkit limits the surface to what that version provides. `cuda-core` gracefully handles missing features by gating higher-level wrappers behind `cfg` where appropriate.

---

## Generated Type Conventions

Bindgen applies standard C-to-Rust mappings:

| C Type | Rust Type |
|--------|-----------|
| `CUresult` | `u32` (enum-like constants) |
| `CUdevice` | `i32` |
| `CUcontext` | `*mut CUctx_st` |
| `CUstream` | `*mut CUstream_st` |
| `CUmodule` | `*mut CUmod_st` |
| `CUfunction` | `*mut CUfunc_st` |
| `CUdeviceptr` | `u64` |
| `CUevent` | `*mut CUevent_st` |
| `CUmemGenericAllocationHandle` | `u64` |
| `CUmemAllocationProp_st` | `#[repr(C)] struct` |
| `CUlaunchAttribute_st` | Opaque byte blob (see above) |

All enums are translated as Rust enums with `#[repr(u32)]` where possible, or as constant modules when the C header uses preprocessor constants.

---

## Safety

Every function in the generated bindings is `unsafe` because the compiler cannot verify:

- Handle validity (dangling `CUmodule`, freed `CUstream`, etc.)
- Context currentness (most driver calls require the owning context to be bound)
- Memory lifetime (device pointers must outlive their uses)
- Stream ordering (async operations must be correctly synchronized)
- Grid/block dimension limits (launch parameters must fit the device)

`cuda-core` exists to wrap these preconditions in safe APIs where feasible. Use the raw bindings only when you need an escape hatch.

---

## Related Crates

| Crate | Role |
|-------|------|
| [`cuda-core`](../cuda-core/) | Safe RAII wrappers over these raw bindings |
| [`cuda-device`](../cuda-device/) | Device-side intrinsics for GPU kernels in Rust |
| [`cuda-host`](../cuda-host/) | Higher-level host abstractions built on cuda-core |
| [`cuda-async`](../cuda-async/) | Async Rust integration with CUDA streams |

---

## License

Licensed under the [LICENSE-NVIDIA](../../LICENSE-NVIDIA) license. This crate contains no hand-written CUDA API declarations — all types and functions are generated from the NVIDIA CUDA Toolkit headers at build time.
