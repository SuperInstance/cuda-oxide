//! Host-side CUDA runtime.
//!
//! `cuda-host` provides low-level host-side operations: device memory management,
//! kernel module loading, and stream-based execution. It wraps the CUDA Driver API
//! with safe Rust abstractions.
//!
//! # Relationship to cuda-core
//!
//! `cuda-core` is the high-level API built on top of `cuda-host`. Most users should
//! prefer `cuda-core` for its ergonomic interface. Use `cuda-host` when you need
//! fine-grained control over the driver API.

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Required for `type_id_u128`'s use of `core::intrinsics::type_id`. The
// intrinsic is the only way to obtain the same 128-bit hash the backend
// uses while keeping the bound at `T: ?Sized` (stable `TypeId::of` would
// force `T: 'static` on every kernel marker — see `type_id.rs` for why
// that would silently reject non-`'static` borrowing closures).
//
// We accept the `internal_features` warning here: cuda-oxide already ships
// `rustc_codegen_cuda` against `rustc_private` and pins a nightly
// toolchain, so the broader project is firmly inside rustc's internal API
// surface. Anyone trying to lift this helper to a stable crate will hit
// the same gate and have to make the same trade-off there.
#![feature(core_intrinsics)]
#![allow(internal_features)]

//! Host-side utilities for CUDA kernel development.
//!
//! This crate provides CPU-side utilities for preparing data and setting up
//! GPU kernel execution. Unlike `cuda-device` which contains device-side primitives,
//! this crate runs entirely on the host.
//!
//! ## Modules
//!
//! - [`embedded`]: Load `#[cuda_module]` artifact bundles embedded in the host
//!   binary (PTX, cubin, NVVM IR, LTOIR)
//! - [`launch`]: Kernel launch traits (`CudaKernel`, `GenericCudaKernel`)
//! - [`ltoir`]: libNVVM + nvJitLink wrappers (`load_kernel_module`, in-memory
//!   `build_cubin_from_nvvm_ir`, `link_ltoir_to_cubin`)
//! - [`tiling`]: Layout transformations for tensor core operations (tcgen05)
//!
//! ## Macros
//!
//! - [`cuda_module`]: Generate a typed embedded-module loader and per-kernel
//!   sync launch methods from an inline kernel module. Enable the `async`
//!   feature for borrowed and owned async launch methods.
//! - [`cuda_launch!`]: Low-level launch macro retained for migration.
//! - `cuda_launch_async!`: Low-level async launch macro retained for
//!   migration when the `async` feature is enabled.
//!
//! ## Usage
//!
//! ```ignore
//! use cuda_device::{kernel, thread, DisjointSlice};
//! use cuda_host::cuda_module;
//! use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
//!
//! #[cuda_module]
//! mod kernels {
//!     use super::*;
//!
//!     #[kernel]
//!     pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { ... }
//! }
//!
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.default_stream();
//! let module = kernels::load(&ctx)?;
//!
//! let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
//! let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
//! let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
//!
//! module.vecadd(
//!     &stream,
//!     LaunchConfig::for_num_elems(N as u32),
//!     &a_dev,
//!     &b_dev,
//!     &mut c_dev,
//! )?;
//!
//! let c_host = c_dev.to_host_vec(&stream)?;
//! ```

/// Load embedded CUDA modules (cubin, PTX, NVVM IR, LTOIR) from the host binary.
pub mod embedded;
/// Kernel launch traits and argument marshalling (`CudaKernel`, `GenericCudaKernel`, etc.).
pub mod launch;
/// LTOIR pipeline: compile NVVM IR + libdevice → cubin via libNVVM and nvJitLink.
pub mod ltoir;
/// Tensor-core tiling layouts (K-major, MN-major) for tcgen05.
pub mod tiling;
/// Stable 128-bit type identifiers for generic kernel PTX naming.
pub mod type_id;

pub use launch::{
    CudaKernel, GenericCudaKernel, HasLength, KernelScalar, ReadOnly, Scalar, WriteOnly,
    push_kernel_device_slice, push_kernel_scalar, read_only_device_buffer_arg,
    writable_device_buffer_arg,
};
/// Re-export of `type_id_u128` for computing backend-compatible type hashes.
pub use type_id::type_id_u128;

/// Async kernel launch traits and helpers (requires `async` feature).
#[cfg(feature = "async")]
pub use launch::{
    KernelSliceArg, KernelSliceArgMut, load_cuda_module_from_async_context,
    load_kernel_module_async, new_async_kernel_launch, new_owned_async_kernel_launch,
    push_async_kernel_scalar, push_async_read_only_device_slice, push_async_writable_device_slice,
    set_async_kernel_cluster_dim,
};

/// Re-export of the `cuda_async` runtime (requires `async` feature).
#[cfg(feature = "async")]
pub use cuda_async;
/// Re-export of async kernel launch handle types (requires `async` feature).
#[cfg(feature = "async")]
pub use cuda_async::launch::{AsyncKernelLaunch, OwnedAsyncKernelLaunch};

/// Re-export of embedded module loading helpers and error types.
pub use embedded::{EmbeddedModuleError, load_embedded_module, load_first_embedded_module};
/// Loads a compiled kernel module by name. Tries `<name>.cubin`, then
/// `<name>.ptx`, and finally falls through to the LTOIR build path
/// (`<name>.ll` plus libdevice → cubin) when cuda-oxide auto-detected
/// CUDA libdevice math intrinsics during the build. Most beginner code
/// never sees the LTOIR path because `vecadd`-style kernels emit `.ptx`
/// directly. See [`ltoir`] for the underlying pipeline and discovery rules.
pub use ltoir::{LtoirError, load_kernel_module};

// Re-export launch macros from cuda-macros for convenience.
/// Re-export of `cuda_macros::cuda_launch` and `cuda_macros::cuda_module` proc macros.
pub use cuda_macros::{cuda_launch, cuda_module};

/// Re-export of [`cuda_macros::cuda_launch_async`].
///
/// Returns a lazy `cuda_async::launch::AsyncKernelLaunch`. Stream assignment is
/// deferred to the scheduling policy -- call `.sync()` to block or `.await` to
/// suspend.
#[cfg(feature = "async")]
pub use cuda_macros::cuda_launch_async;
/// Re-export of tcgen05 tiling utilities and constants.
pub use tiling::{
    TILE_SIZE, k_major_index, mn_major_index, print_layout_indices, to_k_major_f16, to_mn_major_f16,
};
