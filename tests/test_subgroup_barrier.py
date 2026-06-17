"""Phase-4 subgroup barrier on 4 ranks.

Groups exercised: ``[0, 2]``, ``[1, 3]``, ``[0, 1, 3]``. We monkey-patch
``torch.distributed.new_group`` (and the ``_new_group_with_tag`` private
helper) to raise if invoked — gfc must never create a torch process group
for a subgroup.

Run:
    torchrun --standalone --nproc_per_node=4 tests/test_subgroup_barrier.py
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


def _install_new_group_tripwire() -> None:
    def _explode(*args, **kwargs):
        raise AssertionError(
            "gfc must not call dist.new_group / _new_group_with_tag — "
            f"args={args!r} kwargs={kwargs!r}"
        )

    dist.new_group = _explode  # type: ignore[assignment]
    # _new_group_with_tag is private; if it doesn't exist on this torch, skip.
    if hasattr(dist.distributed_c10d, "_new_group_with_tag"):
        dist.distributed_c10d._new_group_with_tag = _explode  # type: ignore[attr-defined]


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()
    assert world == 4, f"test_subgroup_barrier expects nproc=4, got {world}"

    _install_new_group_tripwire()

    config = SymmetricCollectiveConfig(
        max_group_size=4,
        max_collective_bytes=1 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    groups = [
        runtime.register_group((0, 2)),
        runtime.register_group((1, 3)),
        runtime.register_group((0, 1, 3)),
    ]

    iters_per_group = 50
    for g in groups:
        if g.local_index < 0:
            # Non-members must NOT call into the barrier kernel for that group;
            # they participate in the bootstrap barrier between iters below to
            # keep the world stream-aligned for the test loop.
            continue
        for _ in range(iters_per_group):
            runtime.barrier(g)
    runtime.stream.synchronize()
    # World barrier between iterations of distinct subgroup loops would be
    # gratuitous: each rank's epoch counters are independent per group, and
    # the subgroup barriers themselves provide the only required sync. We
    # just stream-sync and then do a world barrier before teardown.
    dist.barrier(group=runtime.world_pg)

    # Cross-check: each member of each group must have advanced exactly
    # iters_per_group epochs for that group.
    for g in groups:
        peek = runtime._epochs.peek(g.group_id)
        if g.local_index >= 0:
            assert peek == iters_per_group, (
                f"rank {runtime.rank} group {g.ranks}: expected epoch "
                f"{iters_per_group}, got {peek}"
            )
        else:
            assert peek == 0, (
                f"rank {runtime.rank} non-member of {g.ranks} but epoch={peek}"
            )

    rank_print(
        "ok subgroups exercised: "
        + ", ".join(f"{g.ranks}={'mem' if g.local_index >= 0 else 'non'}" for g in groups)
    )

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
