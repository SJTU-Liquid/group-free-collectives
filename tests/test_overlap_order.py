"""Phase-8 overlap_order: rank 1 alternates between groups G1 and G2.

Setup: 3 ranks. G1 = (0, 1), G2 = (1, 2). Rank 1 is in both groups; rank 0
only in G1; rank 2 only in G2. Each iteration, every rank issues all_gather
on its group(s), with rank 1 doing G1 then G2 in sequence. 100 iterations.

Verifies that issuing collectives on different subgroups in sequence on
the same rank does not deadlock. The host-side _submit_lock + stream FIFO
order is what makes it correct.

Run:
    torchrun --standalone --nproc_per_node=3 tests/test_overlap_order.py
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


def main() -> int:
    device = setup_nccl()
    assert dist.get_world_size() == 3, "test_overlap_order expects nproc=3"

    config = SymmetricCollectiveConfig(
        max_group_size=3,
        max_collective_bytes=1 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g1 = runtime.register_group((0, 1))
    g2 = runtime.register_group((1, 2))

    n_iter = 100
    n_elems = 1024

    def do_one(g):
        inp = torch.full((n_elems,), runtime.rank, dtype=torch.int32, device=device)
        out = torch.empty(g.size * n_elems, dtype=torch.int32, device=device)
        runtime.all_gather(inp, out, g)
        return inp, out

    for it in range(n_iter):
        if runtime.rank == 0:
            _, out = do_one(g1)
            runtime.stream.synchronize()
            for i, peer in enumerate(g1.ranks):
                seg = out[i * n_elems : (i + 1) * n_elems]
                assert torch.all(seg == peer), f"rank 0 g1 seg {i}"
        elif runtime.rank == 1:
            _, out1 = do_one(g1)
            _, out2 = do_one(g2)
            runtime.stream.synchronize()
            for i, peer in enumerate(g1.ranks):
                seg = out1[i * n_elems : (i + 1) * n_elems]
                assert torch.all(seg == peer), f"rank 1 g1 seg {i}"
            for i, peer in enumerate(g2.ranks):
                seg = out2[i * n_elems : (i + 1) * n_elems]
                assert torch.all(seg == peer), f"rank 1 g2 seg {i}"
        else:
            _, out = do_one(g2)
            runtime.stream.synchronize()
            for i, peer in enumerate(g2.ranks):
                seg = out[i * n_elems : (i + 1) * n_elems]
                assert torch.all(seg == peer), f"rank 2 g2 seg {i}"

        if (it + 1) % 25 == 0:
            rank_print(f"iter {it + 1}/{n_iter} ok")

    rank_print(f"ok overlap_order {n_iter} iters")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
