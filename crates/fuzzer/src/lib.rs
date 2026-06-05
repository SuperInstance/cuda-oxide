/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Differential codegen fuzzer: compares cuda-oxide output vs LLVM.
//!
//! This crate hosts the reusable pieces shared by the rustlantis-style
//! differential tester. The fuzzer compiles the same generated program with
//! both cuda-oxide and standard LLVM backends and compares their trace
//! outputs to detect codegen discrepancies.
//!
//! - A device-compatible single-`u64` trace state, exposed via
//!   [`trace_reset`], [`trace_finish`], and the generic [`dump_var`].
//! - A [`TraceValue`] / [`TraceDump`] pair so generated programs can fold
//!   intermediate values of arbitrary supported scalar types into the trace.
//!
//! The runtime layout (vendored rustlantis source under `rustlantis/`,
//! `tools/mir_generator.py`, etc.) lives next to this library. The library
//! itself is `no_std` and contains nothing CUDA- or std-specific.

#![no_std]

/// FNV-1a trace state and value-dumping traits.
pub mod trace;

pub use trace::{TraceDump, TraceValue, dump_var, trace_finish, trace_reset};
