"""Dynamic per-group rank-list allocation.

Proves the old fixed ``max_groups=64`` cap is gone: a runtime can register far
more than 64 groups, each backed by its own device-resident rank-list tensor,
and ``unregister_group`` frees that tensor and lets the same id be registered
again (the per-group epoch counter stays monotonic, so barrier tokens are
never reused across the unregister/re-register boundary).

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_dynamic_group_alloc.py
    torchrun --standalone --nproc_per_node=8 tests/test_dynamic_group_alloc.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
)

from tests._harness import rank_print, setup_nccl, teardown  # noqa: E402

# Comfortably above the retired 64-group cap.
NUM_GROUPS = 100


def _full_world_gids(world: int) -> list[tuple[int, tuple[int, ...]]]:
    """``NUM_GROUPS`` distinct full-world groups (distinct explicit ids).

    Full-world membership keeps every rank a participant, so a barrier over
    each group is well-formed without per-rank scheduling divergence.
    """
    ranks = tuple(range(world))
    return [(0x5000 + i, ranks) for i in range(NUM_GROUPS)]


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()

    # No max_groups knob — registration is unbounded.
    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=1 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    specs = _full_world_gids(world)

    # 1. Register > 64 groups; each gets a distinct, correct device rank list.
    groups = []
    seen_ptrs = set()
    for gid, ranks in specs:
        g = runtime.register_group(ranks, group_id=gid)
        assert g.local_index >= 0, f"rank should be a member of full-world group {gid}"
        assert g.ranks_dev is not None, "member group must carry a device rank list"
        assert tuple(g.ranks_dev.cpu().tolist()) == ranks, (
            f"device rank list {g.ranks_dev.cpu().tolist()} != {ranks}"
        )
        ptr = int(g.ranks_dev.data_ptr())
        assert ptr not in seen_ptrs, "each group must own a distinct allocation"
        seen_ptrs.add(ptr)
        groups.append(g)

    assert len(runtime._registered) == NUM_GROUPS

    # 2. A barrier on every group must complete (data path works past 64).
    for g in groups:
        runtime.barrier(g)
    runtime.stream.synchronize()

    # 3. A non-member subgroup gets no device tensor (members-only allocation).
    if world >= 3:
        non_member = tuple(r for r in range(world) if r != runtime.rank)[:2]
        ng = runtime.register_group(non_member, group_id=0x9000)
        assert ng.local_index == -1
        assert ng.ranks_dev is None, "non-member descriptor must not allocate"
        runtime.unregister_group(ng)

    # 4. Free every group; the registry empties.
    for g in groups:
        runtime.unregister_group(g)
    assert len(runtime._registered) == 0
    # Freed handles drop their device reference.
    assert all(g.ranks_dev is None for g in groups)

    # 5. Re-register the same ids and barrier again — proves free + realloc;
    #    the per-group epoch resumes monotonically (no token reuse) after free.
    regrps = [runtime.register_group(ranks, group_id=gid) for gid, ranks in specs]
    # Regression guard: re-registration must NOT reset the per-group epoch to
    # 0. Every gid was barriered once before release, so its epoch must have
    # advanced past 0 — a reset would reuse barrier tokens and let a stale
    # signal cell satisfy a fresh barrier (premature completion / data race).
    for gid, _ in specs:
        assert runtime._epochs.peek(gid) >= 1, (
            f"epoch for re-registered gid {gid:#x} reset to 0 — token reuse hazard"
        )
    for g in regrps:
        runtime.barrier(g)
    runtime.stream.synchronize()

    # 6. Idempotent / double-free is a no-op.
    runtime.unregister_group(regrps[0])
    runtime.unregister_group(regrps[0])

    rank_print(f"ok dynamic group alloc world={world} groups={NUM_GROUPS}")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
