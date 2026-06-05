# cuda-core Refactoring Plan

> **Context:** This plan is part of the SuperInstance fork's systems-level hardening effort. `cuda-core` is the foundation of the host runtime; its correctness and documentation directly impact production GPU services and agent runtimes.

## Audit Summary

**Crate purpose:** Safe RAII wrappers around the CUDA driver API.

**Lines of code:** ~3.3K

**Downstream consumers:** `cuda-host`, `cuda-async`, and user code via `CudaContext`, `DeviceBuffer<T>`, etc.

---

## Issues Found

### 1. Missing `// SAFETY:` comments on unsafe blocks

Several `unsafe` blocks lack inline safety comments, which violates Rust unsafe-code guidelines (RFC 2585 spirit) and makes review harder.

- `peer.rs` — `can_access_peer`: `assume_init()` after `cuDeviceCanAccessPeer`
- `vmm.rs` — `set_mem_location_device`: pointer offset write
- `vmm.rs` — `PhysicalAllocation::new`: `cuMemCreate` + `assume_init()`
- `vmm.rs` — `VirtualReservation::new`: `cuMemAddressReserve` + `assume_init()`
- `vmm.rs` — `Mapping::new`: `cuMemMap`
- `vmm.rs` — `PhysicalAllocation::Drop`: `cuMemRelease`
- `vmm.rs` — `VirtualReservation::Drop`: `cuMemAddressFree`
- `vmm.rs` — `Mapping::Drop`: `cuMemUnmap`
- `vmm.rs` — `set_access`: `cuMemSetAccess`
- `vmm.rs` — `allocation_granularity`: `cuMemGetAllocationGranularity` + `assume_init()`
- `module.rs` — `ConstantHandle::write_async_staged`: `drop_staged_bytes` callback

### 2. Missing documentation on public items

- `embedded.rs` — `EmbeddedModule` and its public methods lack `///` docs
- `embedded.rs` — Free functions (`artifact_bundles_from_current_exe`, etc.) lack docs

### 3. API inconsistency (minor)

- `embedded.rs` uses `pub use oxide_artifacts::{...}` but the re-exported types are not documented

---

## Proposed Fixes

### Phase 1: Safety Documentation (P0 — correctness)

1. Add `// SAFETY:` comments to **every** `unsafe` block explaining:
   - Why the preconditions are met (e.g., "`cuDeviceCanAccessPeer` writes exactly one `i32`, so `assume_init()` is sound")
   - What invariant is being maintained (e.g., "`cuMemMap` succeeds because `PhysicalAllocation` size was validated at construction")
2. Audit all `Drop` impls for panic safety — `cuMemRelease` etc. should not panic, but document the behavior if the driver returns an error.

### Phase 2: Public API Documentation (P1 — usability)

1. Add `///` doc comments to all undocumented public items in `embedded.rs`.
2. Ensure `DriverError` formatting helper `_fmt` has consistent documentation.
3. Document VMM and peer-access APIs in `README.md` with usage examples.

### Phase 3: Systems Hardening (P2 — production readiness)

1. **Async stream error propagation** — `CudaStream` currently drops errors on the floor for some async copy paths. Add `CudaEvent`-based error checking or integrate with `cuda-async` error channels.
2. **VMM fragmentation resistance** — `VirtualReservation` currently reserves fixed-size contiguous VA ranges. Evaluate chunked reservation pools for long-running services.
3. **Peer-access refcounting** — `enable_peer_access` is not reference-counted; multiple contexts enabling peer access to the same device can lead to double-disable on drop. Consider an `Arc`-like peer-access handle.
4. **Memory pool integration (CUDA 12.2+)** — Add optional `cuMemPool*` support behind a feature flag for alloc-free performance on Ampere+.

### Phase 4: Testing & Observability (P2 — reliability)

1. Add miri-compatible tests for `DeviceBuffer<T>` layout and alignment.
2. Add stress tests for VMM create/map/unmap cycles.
3. Add tracing / logging integration behind `tracing` feature flag for production debugging.

---

## Status

- [x] Plan written
- [ ] Phase 1: Safety comments applied
- [ ] Phase 2: Docs and README updated
- [ ] Phase 3: Systems hardening (stream errors, VMM pools, peer refcounting)
- [ ] Phase 4: Testing & observability

---

## Related Work

- `ARCHITECTURE.md` § Runtime Architecture — describes how `cuda-core` fits into the host runtime stack
- `crates/cuda-async/README.md` — async layer built on top of `cuda-core`
- `crates/cuda-host/README.md` — typed launch layer consuming `cuda-core` primitives
