# cuda-core Refactoring Plan

## Audit Summary

**Crate purpose:** Safe RAII wrappers around the CUDA driver API.

**Lines of code:** ~3.4K

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

## Proposed Fixes

1. Add `// SAFETY:` comments to every `unsafe` block explaining why the operation is sound.
2. Add `///` doc comments to all undocumented public items in `embedded.rs`.
3. Ensure `DriverError` formatting helper `_fmt` has consistent documentation.
4. Update README.md to mention VMM and peer-access APIs.

## Status

- [x] Plan written
- [ ] Fixes applied
- [ ] README updated
