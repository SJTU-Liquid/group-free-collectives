"""Complex subgroup coverage.

This test is intentionally about subgroup shape, not performance.  It covers
non-contiguous, reversed, overlapping, and full-world groups without creating
torch process groups for those subgroups.  Only ranks that are members of a
case call ``register_group`` or the collective; non-members skip it entirely.

Run:
    torchrun --standalone --nproc_per_node=4 tests/test_complex_groups.py
    torchrun --standalone --nproc_per_node=8 tests/test_complex_groups.py
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
            "gfc must not call dist.new_group / _new_group_with_tag for subgroups; "
            f"args={args!r} kwargs={kwargs!r}"
        )

    dist.new_group = _explode  # type: ignore[assignment]
    if hasattr(dist.distributed_c10d, "_new_group_with_tag"):
        dist.distributed_c10d._new_group_with_tag = _explode  # type: ignore[attr-defined]


def _group_cases(world: int) -> list[tuple[int, ...]]:
    """Return deterministic subgroup cases valid for ``world`` ranks."""
    assert 2 <= world <= 8
    cases: list[tuple[int, ...]] = [
        tuple(range(world)),
        tuple(reversed(range(world))),
        (0, world - 1),
    ]
    if world >= 3:
        cases.extend(
            [
                (0, 2),
                (1, 2),
                tuple(reversed(range(1, world))),
            ]
        )
    if world >= 4:
        cases.extend(
            [
                (0, 2, 3),
                (1, 3),
                (3, 1, 0),
            ]
        )
    if world >= 5:
        cases.extend(
            [
                (0, 2, 4),
                (4, 2, 1, 0),
            ]
        )
    if world >= 6:
        cases.append((0, 2, 5))
    if world >= 8:
        cases.extend(
            [
                (7, 6, 4, 1),
                (3, 0, 7, 2, 5),
                (6, 2, 4, 0),
                (7, 5, 3, 1),
            ]
        )

    deduped: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for ranks in cases:
        if ranks not in seen:
            seen.add(ranks)
            deduped.append(ranks)
    return deduped


def _ag_pattern(rank: int, case_idx: int, nbytes: int, device) -> torch.Tensor:
    idx = torch.arange(nbytes, dtype=torch.int64, device=device)
    return (((case_idx + 1) * 29 + (rank + 1) * 17 + idx * 7) & 0xFF).to(torch.uint8)


def _a2a_pattern(sender: int, dst_local_idx: int, case_idx: int, nbytes: int, device) -> torch.Tensor:
    idx = torch.arange(nbytes, dtype=torch.int64, device=device)
    return (
        ((case_idx + 1) * 43 + (sender + 1) * 19 + (dst_local_idx + 1) * 11 + idx * 5)
        & 0xFF
    ).to(torch.uint8)


def _run_all_gather(runtime: SymmetricCollectiveRuntime, group, case_idx: int) -> None:
    nbytes = 4096 + case_idx * 257
    inp = _ag_pattern(runtime.rank, case_idx, nbytes, runtime.device)
    out = torch.empty(group.size * nbytes, dtype=torch.uint8, device=runtime.device)

    runtime.all_gather(inp, out, group)
    runtime.stream.synchronize()

    for seg_idx, peer in enumerate(group.ranks):
        expected = _ag_pattern(peer, case_idx, nbytes, runtime.device)
        got = out[seg_idx * nbytes : (seg_idx + 1) * nbytes]
        if not torch.equal(got, expected):
            mismatch = (got != expected).nonzero(as_tuple=False)
            first = int(mismatch[0, 0]) if mismatch.numel() else -1
            raise AssertionError(
                f"all_gather mismatch case={case_idx} group={group.ranks} "
                f"seg={seg_idx} peer={peer} first_bad={first}"
            )


def _run_all2all(runtime: SymmetricCollectiveRuntime, group, case_idx: int) -> None:
    slice_bytes = 2048 + case_idx * 131
    inp = torch.empty(group.size * slice_bytes, dtype=torch.uint8, device=runtime.device)
    for dst_local_idx in range(group.size):
        inp[dst_local_idx * slice_bytes : (dst_local_idx + 1) * slice_bytes] = _a2a_pattern(
            runtime.rank, dst_local_idx, case_idx, slice_bytes, runtime.device
        )

    out = torch.empty_like(inp)
    runtime.all2all(inp, out, group)
    runtime.stream.synchronize()

    for seg_idx, peer in enumerate(group.ranks):
        expected = _a2a_pattern(peer, group.local_index, case_idx, slice_bytes, runtime.device)
        got = out[seg_idx * slice_bytes : (seg_idx + 1) * slice_bytes]
        if not torch.equal(got, expected):
            mismatch = (got != expected).nonzero(as_tuple=False)
            first = int(mismatch[0, 0]) if mismatch.numel() else -1
            raise AssertionError(
                f"all2all mismatch case={case_idx} group={group.ranks} "
                f"seg={seg_idx} peer={peer} first_bad={first}"
            )


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()
    assert 2 <= world <= 8, f"test_complex_groups expects nproc in [2, 8], got {world}"

    _install_new_group_tripwire()

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=4 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    cases = _group_cases(world)
    for case_idx, ranks in enumerate(cases):
        if runtime.rank in ranks:
            group = runtime.register_group(ranks)
            assert group.local_index >= 0
            _run_all_gather(runtime, group, case_idx)
            _run_all2all(runtime, group, case_idx)

        # Keep this test focused on membership/local-registration semantics.
        # Interleaved overlapping subgroup scheduling is covered separately.
        dist.barrier(group=runtime.world_pg)

    rank_print(
        f"ok complex groups world={world} cases={len(cases)} "
        "with member-only registration"
    )

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
