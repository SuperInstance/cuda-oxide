/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVVM dialect — GPU intrinsic operations for LLVM's NVPTX backend.
//!
//! `dialect-nvvm` represents NVIDIA GPU intrinsics as Pliron IR operations.
//! Each op in this dialect corresponds to one or more LLVM NVVM intrinsics
//! (or inline PTX assembly), and is lowered to the LLVM dialect by `mir-lower`.
//!
//! # Dialect Position in the Pipeline
//!
//! ```text
//! dialect-mir intrinsic call (e.g. MirIntrinsicOp for thread_id_x)
//!       │  ← mir-importer translates GPU intrinsics here
//!       ▼
//! dialect-nvvm op (e.g. ReadPtxSregTidXOp)   ← THIS DIALECT
//!       │  ← mir-lower emits LLVM intrinsic call / inline PTX
//!       ▼
//! llvm.call @llvm.nvvm.read.ptx.sreg.tid.x()
//! ```
//!
//! # Operation Categories
//!
//! | Module       | Description                     | Arch       |
//! |--------------|---------------------------------|------------|
//! | `thread`     | Thread/block/grid index reads   | All        |
//! | `warp`       | Shuffle, vote                   | All        |
//! | `atomic`     | Atomic load/store/RMW/CAS       | sm_70+     |
//! | `cluster`    | Thread-block-cluster + DSMEM    | Hopper+    |
//! | `mbarrier`   | Async hardware barriers         | Hopper+    |
//! | `tma`        | Tensor Memory Accelerator       | Hopper+    |
//! | `wgmma`      | Warpgroup Matrix Multiply-Acc   | Hopper only|
//! | `tcgen05`    | Tensor Core Gen 5               | Blackwell+ |
//! | `stmatrix`   | Shared-mem matrix store         | Hopper+    |
//! | `clc`        | CLC cooperative launching       | sm_90+     |
//! | `debug`      | `printf` / assertion support    | All        |

/// NVVM dialect operations — GPU intrinsics for thread indexing, warp shuffle, atomics,
/// TMA, wgmma, cluster barriers, debug printf, and more.
pub mod ops;

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

/// Name of the NVVM dialect in pliron (`"nvvm"`).
pub const NVVM_DIALECT_NAME: &str = "nvvm";

/// Register all NVVM dialect ops with the given context.
pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::new(NVVM_DIALECT_NAME));

    ops::register(ctx);
}
