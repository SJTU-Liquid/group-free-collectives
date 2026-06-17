"""Phase-8 cross-kernel publication test.

Per iter: stage *random* bytes into comm_buf via a separate kernel (copy_),
then run all_gather (pre-barrier kernel, then pull kernel). The barrier's
release/acquire-sys semantics must make the staging-kernel's stores visible
to the peer's pull-kernel even though they live in *distinct* kernel
launches on the same stream.

2000 iters x 2 ranks; randomized inputs so accidental zeros from a previous
iter cannot mask a bug.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_cross_kernel_publication.py
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

    n_iter = 2000
    n_elems = 4096  # 16 KiB at fp32
    dtype = torch.float32

    # Use a per-rank deterministic RNG so peers can recompute each other's
    # expected segment without any cross-rank communication.
    gen = torch.Generator(device=device)
    gen.manual_seed(0x1357 + runtime.rank)
    peer_gen = torch.Generator(device=device)
    peer_gen.manual_seed(0x1357 + (1 - runtime.rank))

    for it in range(n_iter):
        inp = torch.empty(n_elems, dtype=dtype, device=device)
        inp.normal_(generator=gen)
        out = torch.empty(2 * n_elems, dtype=dtype, device=device)
        runtime.all_gather(inp, out, g)
        runtime.stream.synchronize()

        # The peer's segment is the input the peer would have generated with
        # its generator on the same iteration. Both sides advance their own
        # generator by n_elems normal samples per iteration, so peer_gen
        # tracks the peer's pattern.
        peer_inp = torch.empty(n_elems, dtype=dtype, device=device)
        peer_inp.normal_(generator=peer_gen)

        # Output layout: segment i corresponds to group.ranks[i]. group is
        # (0, 1), so segment 0 is rank 0's input and segment 1 is rank 1's.
        own_seg = out[runtime.rank * n_elems : (runtime.rank + 1) * n_elems]
        peer_seg = out[(1 - runtime.rank) * n_elems : (2 - runtime.rank) * n_elems]

        if not torch.equal(own_seg, inp):
            mismatch = (own_seg != inp).nonzero(as_tuple=False)
            first = int(mismatch[0, 0])
            raise AssertionError(
                f"iter {it}: own segment mismatch at {first}"
            )
        if not torch.equal(peer_seg, peer_inp):
            mismatch = (peer_seg != peer_inp).nonzero(as_tuple=False)
            first = int(mismatch[0, 0])
            raise AssertionError(
                f"iter {it}: peer segment mismatch at {first} "
                f"(got={float(peer_seg[first]):.6g} "
                f"exp={float(peer_inp[first]):.6g})"
            )

        if (it + 1) % 500 == 0:
            rank_print(f"iter {it + 1}/{n_iter} ok")

    rank_print(f"ok cross-kernel publication: {n_iter} all_gather iters")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
