"""Phase-5 all_gather test on 2 or 4 ranks.

Each rank fills an input tensor with its own rank id, runs all_gather, then
checks that ``output[i * input.numel() : (i+1) * input.numel()]`` is all
``group.ranks[i]``. Cycles through fp16/bf16/fp32/uint8.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_all_gather.py
    torchrun --standalone --nproc_per_node=4 tests/test_all_gather.py
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


_DTYPES = [torch.uint8, torch.float32, torch.bfloat16, torch.float16]


def _one_case(runtime, group, dtype: torch.dtype, n_elems: int) -> None:
    inp = torch.full(
        (n_elems,), runtime.rank, dtype=dtype, device=runtime.device
    )
    out = torch.empty(
        group.size * n_elems, dtype=dtype, device=runtime.device
    )

    runtime.all_gather(inp, out, group)
    runtime.stream.synchronize()

    for i in range(group.size):
        seg = out[i * n_elems : (i + 1) * n_elems]
        expected_rank = group.ranks[i]
        # Compare as raw bytes — robust across all dtypes.
        ref = torch.full(
            (n_elems,), expected_rank, dtype=dtype, device=runtime.device
        )
        if not torch.equal(seg, ref):
            mismatches = (seg != ref).nonzero(as_tuple=False)
            first = int(mismatches[0, 0]) if mismatches.numel() else -1
            raise AssertionError(
                f"all_gather mismatch dtype={dtype} n={n_elems} "
                f"segment {i} (rank={expected_rank}) first_bad_idx={first} "
                f"got={float(seg[max(first,0)]):.6g} exp={expected_rank}"
            )


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()
    assert world in (2, 4), f"test_all_gather supports nproc 2 or 4, got {world}"

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=4 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g_all = runtime.register_group(tuple(range(world)))

    for dtype in _DTYPES:
        for n_elems in (16, 1024, 16 * 1024):
            _one_case(runtime, g_all, dtype, n_elems)

    # Also exercise a subgroup (reversed order) on 4 ranks.
    if world == 4:
        g_rev = runtime.register_group(tuple(reversed(range(world))))
        for dtype in (torch.float32, torch.bfloat16):
            _one_case(runtime, g_rev, dtype, 1024)

    rank_print(f"ok all_gather across {world} ranks, all dtypes")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
