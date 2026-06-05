/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU intrinsic dispatch and expansion.
//!
//! This module handles the translation of `cuda_device` intrinsic calls into
//! `dialect-nvvm` operations. Intrinsics are organized by functional category:
//!
//! | Module      | Intrinsics                                                                   |
//! |-------------|------------------------------------------------------------------------------|
//! | `indexing`  | `threadIdx_*`, `blockIdx_*`, `index_1d`, `index_2d::<S>`, `index_2d_runtime` |
//! | `sync`      | `sync_threads`, `mbarrier_*`, `fence_*`                                      |
//! | `cluster`   | `cluster_ctaidX`, `cluster_sync`, `map_shared_rank`                          |
//! | `warp`      | `shuffle_*`, `vote_*`, `lane_id`                                             |
//! | `wgmma`     | Hopper WGMMA matrix operations                                               |
//! | `tcgen05`   | Blackwell tensor core (tcgen05) operations                                   |
//! | `tma`       | Tensor Memory Access (TMA) operations                                        |
//! | `memory`    | `SharedArray`, `stmatrix_*`, type conversions                                |
//! | `debug`     | `clock`, `clock64`, `globaltimer`, `trap`, `breakpoint`                      |
//!
//! # Architecture
//!
//! Each intrinsic module exports `emit_*` functions that:
//! 1. Take MIR operands and translate them to pliron IR values
//! 2. Create the appropriate `dialect-nvvm` operations
//! 3. Store results in the value map
//! 4. Emit a zero-operand `mir.goto` to the call's single successor block
//!
//! # Note
//!
//! Currently, all emit functions remain in `terminator/mod.rs` for compilation
//! stability. This module structure is prepared for gradual migration of
//! functions to their respective category modules.

// Submodules for intrinsic categories (to be populated incrementally)

/// `std::intrinsics::atomic_*` and `core::sync::atomic` operations.
pub mod atomic;

/// `core::intrinsics::bit_*` and `ctlz`/`cttz`/`popcount` operations.
pub mod bitops;

/// Cooperative-launch grid-level collective operations.
pub mod clc;

/// Thread-block cluster intrinsics (`cluster_ctaid_*`, `cluster_sync`, …).
pub mod cluster;

/// Debug intrinsics: `prof_trigger`, `vprintf`, `trap`, `breakpoint`, clocks.
pub mod debug;

/// Rust float math intrinsics (`sin`, `cos`, `exp`, `pow`, …) → `__nv_*` libdevice calls.
pub mod float_math;

/// Thread/block indexing intrinsics (`threadIdx_*`, `blockIdx_*`, `index_1d`, `index_2d`).
pub mod indexing;

/// Shared-memory and pointer intrinsics (`SharedArray`, `DynamicSharedArray`, `stmatrix`).
pub mod memory;

/// Saturating arithmetic intrinsics (`saturating_add`, `saturating_sub`).
pub mod saturating;

/// Synchronization intrinsics (`sync_threads`, `mbarrier_*`, `fence_*`).
pub mod sync;

/// Blackwell tcgen05 tensor-core operations (TMEM, MMA, commit, cp).
pub mod tcgen05;

/// Tensor Memory Access (TMA) bulk-copy intrinsics.
pub mod tma;

/// Warp-level primitives (`shfl_*`, `vote_*`, `lane_id`).
pub mod warp;

/// Hopper WGMMA (warpgroup MMA) matrix-multiply intrinsics.
pub mod wgmma;
