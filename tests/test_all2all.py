"""Phase-6 all2all test on 2 or 4 ranks.

Pattern: rank ``s`` fills slice ``t`` of its input with a deterministic byte
pattern over ``(s, t, byte_offset)``. After all2all, every rank verifies
that slice ``i`` of its output matches the pattern for
``(group.ranks[i], group.local_index, byte_offset)`` — i.e. the slice that
peer ``group.ranks[i]`` had destined for *this* rank.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_all2all.py
    torchrun --standalone --nproc_per_node=4 tests/test_all2all.py
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


def _slice_pattern(s: int, t: int, slice_bytes: int, device) -> torch.Tensor:
    """Bytes for slice (sender=s, dest_local=t)."""
    idx = torch.arange(slice_bytes, dtype=torch.int64, device=device)
    p = ((s + 1) * 1009 + (t + 1) * 31 + idx * 17) & 0xFF
    return p.to(torch.uint8)


def _one_case(runtime, group, slice_bytes: int) -> None:
    inp = torch.empty(group.size * slice_bytes, dtype=torch.uint8, device=runtime.device)
    for t in range(group.size):
        inp[t * slice_bytes : (t + 1) * slice_bytes] = _slice_pattern(
            runtime.rank, t, slice_bytes, runtime.device
        )

    out = torch.empty(group.size * slice_bytes, dtype=torch.uint8, device=runtime.device)
    runtime.all2all(inp, out, group)
    runtime.stream.synchronize()

    for i in range(group.size):
        peer = group.ranks[i]
        # The slice that peer sent to *us* — peer's local_index for us is
        # our local_index in the group.
        expected = _slice_pattern(peer, group.local_index, slice_bytes, runtime.device)
        got = out[i * slice_bytes : (i + 1) * slice_bytes]
        if not torch.equal(got, expected):
            mismatch = (got != expected).nonzero(as_tuple=False)
            first = int(mismatch[0, 0]) if mismatch.numel() else -1
            raise AssertionError(
                f"all2all mismatch slice_bytes={slice_bytes} segment={i} "
                f"peer={peer} first_bad={first} "
                f"got=0x{int(got[first]):02x} exp=0x{int(expected[first]):02x}"
            )


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()
    assert world in (2, 4), f"test_all2all supports nproc 2 or 4, got {world}"

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=4 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g_all = runtime.register_group(tuple(range(world)))

    for slice_bytes in (16, 1024, 64 * 1024, 256 * 1024):
        _one_case(runtime, g_all, slice_bytes)

    # Subgroup test on 4 ranks: [0, 1, 3] (non-contiguous, asymmetric size).
    if world == 4:
        g_sub = runtime.register_group((0, 1, 3))
        if g_sub.local_index >= 0:
            for slice_bytes in (1024, 64 * 1024):
                _one_case(runtime, g_sub, slice_bytes)

    rank_print(f"ok all2all across {world} ranks")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
