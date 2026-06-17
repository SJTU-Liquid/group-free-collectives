# Dynamic per-group rank-list allocation

**Date:** 2026-06-13
**Status:** Approved (brainstorm)

## Problem

`SymmetricCollectiveRuntime` caps the number of registered groups at
`config.max_groups` (default **64**). At init it preallocates a device tensor
`_group_ranks_pool` of shape `(max_groups, max_group_size)` and hands out rows
from a freelist (`_free_pool_rows`). `register_group` pops a row, writes the
ordered rank list into it, and stores the row index in
`GroupDescriptor.pool_row`. Groups are **never freed** — there is no
`unregister`/release path, so the freelist only ever shrinks.

This is unreasonable: a group only needs a small device-resident rank list
(`uint32[group_size]`). There is no reason to cap the count or to pay for a
64-row table up front.

## Goal

Allocate each group's rank list on demand at `register_group`, store it in the
group handle, and free it at `unregister_group`. Remove the `max_groups` cap
and the preallocated pool entirely.

## Design

### `GroupDescriptor` (gfc/group.py)

- Replace `pool_row: int` with `ranks_dev` — the per-group `uint32[group_size]`
  device tensor holding the ordered rank list, declared as
  `field(default=None, compare=False, repr=False)`:
  - `compare=False` keeps the frozen dataclass hashable / equatable on its
    identity fields only (`group_id, ranks, local_index, size`); a `torch.Tensor`
    is neither hashable nor sanely `==`-comparable.
  - `default=None` lets pure-Python construction (unit tests) and **non-member**
    descriptors carry no tensor.
  - `repr=False` avoids dumping tensor contents in logs.
- group.py keeps no hard `torch` import — `ranks_dev` is typed as an opaque
  reference (`object` / `TYPE_CHECKING`-only annotation). Docstring updated to
  note the handle now owns a device tensor.

### Runtime (gfc/runtime.py)

- **Remove** `_group_ranks_pool`, `_free_pool_rows`, `_allocate_pool_row`, and
  the `assert len(self._registered) < self.config.max_groups` check.
- `register_group`:
  - Allocate `t = torch.empty(group_size, dtype=torch.uint32, device=self.device)`
    and copy the ranks in (single tiny H2D).
  - Size is **exactly `group_size`** — no sentinel padding. Verified safe:
    every kernel reads only `group_ranks[0:group_size]` (`barrier`, `all_gather`,
    `all2all`, `tma_paths` index `peer_idx < GROUP_SIZE`; `fused` masks lanes
    `peer_lane >= GROUP_SIZE` with `other=0`).
  - Allocate **only for members** (`local_index >= 0`). Non-members never launch
    a kernel that reads the row, so their `ranks_dev` stays `None`.
  - Store `t` in the descriptor.
- New `unregister_group(group)`:
  - `pop` the entry from `self._registered` (no-op if absent → idempotent /
    safe double-free).
  - If the descriptor holds a tensor: `t.record_stream(self.stream)` then clear
    the reference (`object.__setattr__(desc, "ranks_dev", None)`).
    `record_stream` makes the caching allocator defer reclaiming the block until
    in-flight collectives already queued on `self.stream` drain past this point
    — so "free immediately" is safe against queued kernels **without a blocking
    `synchronize()`**.
  - Keep the per-group epoch entry (`self._epochs`) so a later re-registration
    of the same `group_id` continues monotonically and cannot reuse recent
    barrier tokens.
- `_group_ranks_row_ptr(desc)` returns `int(desc.ranks_dev.data_ptr())`.

### Config (gfc/config.py)

- Remove the `max_groups` field and its `assert self.max_groups > 0`.

### Call sites to update

- `tests/test_skeleton_unit.py` — `pool_row=0` → `ranks_dev=None` (3 sites).
- `tests/test_barrier_stress.py`, `tests/test_complex_groups.py`,
  `benchmarks/bench_multigroup_collectives.py` — drop the `max_groups=` kwarg.

## Testing

- Unit (CPU): `GroupDescriptor` constructs with `ranks_dev=None`; identity
  equality/hash unaffected by the tensor field.
- GPU (multi-rank, run via the project venv): register **> 64** groups (e.g. 200)
  and run a barrier/collective to prove the old cap is gone; `unregister_group`
  then re-register the same ranks and confirm it works and frees the tensor.

## Trade-offs

Old: one `64×8` alloc up front, O(1) row lookup, hard cap, never frees.
New: one tiny `uint32[group_size]` alloc per live group, unbounded count,
deterministic free. Cost is a small `torch.empty` + H2D per `register_group`,
which is a control-plane op, not on the collective hot path.

## Non-goals

- No context-manager sugar (explicit `unregister_group` only).
- Register/unregister are control-plane ops, not collective hot-path calls.
