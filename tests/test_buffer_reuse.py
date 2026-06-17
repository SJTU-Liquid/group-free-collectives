"""Phase-8 buffer_reuse: one comm buffer, many collectives.

Verifies no stale bytes from a previous collective leak into the next
collective when the single comm buffer is reused.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_buffer_reuse.py
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
    assert dist.get_world_size() == 2

    config = SymmetricCollectiveConfig(
        max_group_size=2,
        max_collective_bytes=1 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g = runtime.register_group((0, 1))

    n_iter = 200
    sizes = [16, 1024, 8 * 1024, 64 * 1024]

    for it in range(n_iter):
        n_elems = sizes[it % len(sizes)]
        # Make each iter's pattern distinct so stale bytes would visibly
        # mismatch.
        marker = (it + 1) & 0xFF
        inp = torch.full((n_elems,), marker, dtype=torch.uint8, device=device) ^ runtime.rank
        out = torch.empty(2 * n_elems, dtype=torch.uint8, device=device)
        runtime.all_gather(inp, out, g)
        runtime.stream.synchronize()

        for i, peer in enumerate(g.ranks):
            seg = out[i * n_elems : (i + 1) * n_elems]
            exp_byte = (marker ^ peer) & 0xFF
            if not torch.all(seg == exp_byte):
                first = int((seg != exp_byte).nonzero(as_tuple=False)[0, 0])
                raise AssertionError(
                    f"iter {it} n={n_elems} seg {i} byte {first} got=0x"
                    f"{int(seg[first]):02x} exp=0x{exp_byte:02x}"
                )

        if (it + 1) % 50 == 0:
            rank_print(f"iter {it + 1}/{n_iter} ok")

    rank_print(f"ok buffer_reuse: {n_iter} all_gather iters")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
