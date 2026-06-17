# gfc — Design overview

This overview is the maintained source of truth for the current protocol. The
original full design spec —
[`specs/2026-05-12-symmetric-collectives-design.md`](./specs/2026-05-12-symmetric-collectives-design.md)
— is **partially historical** (see its Status banner for the mechanisms that
have since changed); dynamic per-group rank-list allocation is specified in
[`specs/2026-06-13-dynamic-group-alloc-design.md`](./specs/2026-06-13-dynamic-group-alloc-design.md).

## Mental model

1. **One symmetric rendezvous on the world group**, never per-subgroup.
2. Symmetric regions, allocated once and never reallocated: `comm_buf` (data
   plane), `signal_buf` (the per-edge barrier token grid), and `step_pad` (the
   per-(channel, chunk, producer) completion grid for the fused path).
3. Subgroups are host-side state plus one small device tensor — a
   `GroupDescriptor` carries an ordered rank tuple, a stable `group_id` (hash
   of the ordered tuple), a `local_index`, and `ranks_dev`, a device-resident
   `uint32[size]` ordered rank list allocated per group at `register_group`
   and freed at `unregister_group`. There is no preallocated pool and no fixed
   cap on the number of live groups; a non-member descriptor carries no device
   tensor (members-only allocation).
4. **Tokens** are 64-bit hashes of `(session_nonce, group_id, epoch)`.
   Pre/post collective barriers use adjacent epochs `(k, k+1)`; a bare
   `barrier` advances by 1; the fused path consumes three epochs (pre, the
   in-kernel step-counter base, and post).
5. **Per-edge double-buffered signalling**. `signal_buf` is laid out per rank
   as `uint64[2 slots, world_size src]`. For each ordered pair `(self, peer)`,
   barrier *N* uses `slot = edge_seq[peer] & 1` (a local, GPU-resident counter)
   and then bumps it. There is **no** separate finish/ack lane: slot reuse is
   delayed by one barrier, and pairwise stream-FIFO ordering guarantees the
   peer consumed the previous value before the slot is reused.
6. **One comm buffer per rank**. Runtime submissions are serialized onto one
   CUDA stream; the post barrier protects comm-buffer reuse. The fused path
   issues this post barrier too — its in-kernel step counters prove *this*
   rank finished its own pulls, not that peers finished reading this rank's
   `comm_buf`.
7. **Release/acquire-sys barrier**: each rank publishes its token into the
   peer's incoming cell with `st.global.release.sys.b64`, then spins on its own
   reciprocal cell with `ld.global.acquire.sys.b64` until it observes the
   peer's token. Exactly one writer per cell per token, so no finish/ack phase
   is needed.

## Data flow for a generic pull collective

```
runtime.all_gather(inp, out, g):
    take a (pre_epoch, post_epoch) pair from _EpochManager
    -- on runtime.stream, under the submit lock:
       1) copy_(inp) into comm_buf                         [stage copy kernel]
       2) barrier_kernel(pre_token)                  [per-edge release/acquire]
       3) all_gather_pull_kernel: each peer's comm_buf -> out [vec or TMA path]
       4) barrier_kernel(post_token)                 [per-edge release/acquire]
```

`p2p_put` follows the same shape but uses `launch_p2p_put_copy` (which is
`launch_pull_copy` with src=local, dst=peer) between the barriers; the
sender owns the data movement. `p2p_get` keeps `launch_pull_copy` with
src=peer, dst=local on the receiver; the sender just stages and barriers.

## TMA path (gated)

`tma_probe.probe_tma_supported` validates that a Triton 1D tensor descriptor
built from a peer pointer actually loads the correct bytes. On hardware
where the probe passes, `runtime.all_gather` uses `_all_gather_tma_kernel`
which builds the descriptor inside the kernel and uses
`tl.load_tensor_descriptor` — TMA hardware does the load. Per-call
gates (16-byte slice/output alignment and aligned tile size) ensure no
half-issued collective; unsatisfied alignment raises `TMARequirementError`
before epochs are consumed.

The probe failing is acceptable (the spec acknowledges this is unverified on
some NVLink P2P setups). `use_tma=False` keeps the runtime fully functional
on the vec path.

## Hard correctness invariants

These are enforced by code or asserts in the runtime:

1. No `dist.new_group` — `test_subgroup_barrier.py` installs a tripwire.
2. No `torch.cuda.synchronize()` in library code.
3. A one-shot collective consumes two epochs `(k, k+1)`; a bare barrier
   consumes one; the fused path consumes three (pre, step-base, post). Epochs
   are strictly monotonic per `group_id` and never reused within a session.
4. Signals are addressed by source global rank, never by group-local index:
   each writer publishes its token into `[slot, self_global_rank]` of the
   peer's grid, where `slot` is the `edge_seq[peer]` parity for that edge.
5. `group_id` hashes the ordered rank list.
6. All submissions onto `runtime.stream` are serialized by the host submit
   lock.
7. No silent fallbacks (TMA is the only optional path and it raises if
   requested but unsupported).
8. CUDA Graph replay is not a supported runtime API in v1; graph capture in
   benchmarks is an explicit experiment, not a correctness contract.

## Test coverage

```
tests/test_skeleton_unit.py             — Phase 1 unit tests (single-process)
tests/test_init.py                       — symm-mem rendezvous, peer ptrs, nonce
tests/test_remote_copy_kernel.py        — standalone pull_copy_kernel
tests/test_barrier_repeated.py          — 1000 bare barriers
tests/test_subgroup_barrier.py          — subgroup barriers, no new_group
tests/test_all_gather.py                — all_gather, dtypes, subgroups
tests/test_all2all.py                   — all2all, dtypes, [0,1,3] subgroup
tests/test_p2p.py                       — p2p_put + p2p_get, 100 epochs each
tests/test_cross_kernel_publication.py — release/acquire-sys across kernels
tests/test_overlap_order.py             — multi-subgroup interleave, no deadlock
tests/test_buffer_reuse.py              — single comm buffer, 200 iters
tests/test_perf_smoke.py                — gfc vs NCCL reference timings
tests/test_tma_probe.py                 — TMA probe + TMA all_gather verify
tests/test_dynamic_group_alloc.py       — per-group rank-list alloc/free, >64 groups
tests/test_review_fixes.py              — arg validation, pipelined alignment, fused reuse
```
