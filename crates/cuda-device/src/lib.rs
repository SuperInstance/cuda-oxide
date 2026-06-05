//! Device-side GPU programming abstractions.
//!
//! `cuda-device` provides the programming model for code that runs **on the GPU**.
//! It exposes thread indexing, shared memory, warp-level primitives, atomics,
//! barriers, cooperative groups, tensor core operations (wgmma, tcgen05), and
//! TMA (Tensor Memory Accelerator) — all in safe, idiomatic Rust.
//!
//! # Programming Model
//!
//! GPU code runs in a SIMT (Single Instruction, Multiple Thread) model:
//!
//! ```text
//! Grid (dim3 gridDim)
//!  └── Block (dim3 blockIdx, dim3 blockDim)
//!       └── Thread (1D/2D/3D index)
//! ```
//!
//! # Key Modules
//!
//! | Module              | Purpose                                    |
//! |---------------------|--------------------------------------------|
//! | `thread`            | Thread indexing (1D, 2D, 3D)              |
//! | `shared`            | Shared memory management                   |
//! | `atomic`            | Scoped atomic operations                   |
//! | `barrier`           | Block-level synchronization                |
//! | `warp`              | Warp shuffle, vote, reduce                 |
//! | `cooperative_groups`| Cooperative group abstractions              |
//! | `wgmma`             | Warp Group Matrix Multiply Accumulate      |
//! | `tcgen05`           | Tensor Core generation 0.5 operations      |
//! | `tma`               | Tensor Memory Accelerator                  |
//!
//! # Relationship to Other Crates
//!
//! - **cuda-core**: Host-side API that *launches* kernels defined with cuda-device
//! - **cuda-macros**: `#[kernel]` attribute that marks functions for GPU compilation
//! - **cuda-async**: Async version of kernel launch and memory operations
//!
//! Code in this crate runs on the device. It is compiled to PTX by `rustc-codegen-cuda`.

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]
#![no_std]

pub use cuda_macros::{
    cluster_launch, constant, convergent, cuda_module, device, gpu_printf, kernel, launch_bounds,
    pure, readonly,
};

// Re-export for convenience
pub mod atomic;
pub mod barrier;
pub mod clc;
pub mod cluster;
pub mod constant;
pub mod cooperative_groups;
pub mod cusimd;
pub mod debug;
pub mod disjoint;
pub mod fence;
pub mod grid;
pub mod shared;
pub mod tcgen05;
pub mod thread;
pub mod tma;
pub mod warp;
pub mod wgmma;

pub use barrier::{
    // Core type
    Barrier,
    BarrierToken,
    GeneralBarrier,
    Invalidated,
    // Typestate managed barrier
    ManagedBarrier,
    MmaBarrier,
    MmaBarrierHandle,
    Ready,
    // Kind markers
    TmaBarrier,
    TmaBarrier0,
    TmaBarrier1,
    // Type aliases
    TmaBarrierHandle,
    // State markers
    Uninit,
};
pub use constant::{ConstantMemory, ConstantMemoryValue};
pub use cusimd::{CuSimd, Float2, Float4, TmemRegs4, TmemRegs32};
pub use disjoint::DisjointSlice;
pub use fence::*;
pub use shared::{DynamicSharedArray, SharedArray};
pub use tcgen05::{
    TensorMemoryHandle, TmemAddress, TmemDeallocated, TmemF32x4, TmemF32x32, TmemGuard, TmemReady,
    TmemUninit,
};
pub use thread::*;
pub use tma::TmaDescriptor;
