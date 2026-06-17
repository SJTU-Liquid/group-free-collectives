"""Phase-2 multi-process init test.

Validates symm-mem rendezvous, non-zero ``buffer_ptrs_dev``, and cross-rank
size agreement. Run via:

    torchrun --standalone --nproc_per_node=2 tests/test_init.py
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist

# Make repo importable when run with `torchrun tests/test_init.py`.
import os as _os
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

from gfc import SymmetricCollectiveConfig, SymmetricCollectiveRuntime  # noqa: E402

from tests._harness import setup_nccl, teardown, rank_print  # noqa: E402


def main() -> int:
    device = setup_nccl()
    config = SymmetricCollectiveConfig(
        max_group_size=min(8, dist.get_world_size()),
        # Trim defaults so tiny CI runs do not allocate 512 MiB/rank.
        max_collective_bytes=4 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    # Non-zero peer pointer tables.
    cb_ptrs = runtime.comm_buf_ptrs_dev
    sb_ptrs = runtime.signal_buf_ptrs_dev
    assert cb_ptrs != 0, f"rank {runtime.rank}: comm_buf_ptrs_dev is null"
    assert sb_ptrs != 0, f"rank {runtime.rank}: signal_buf_ptrs_dev is null"

    # Local-side data_ptr of each region is also non-zero.
    assert runtime.regions.comm_buf.local_ptr != 0
    assert runtime.regions.signal_buf.local_ptr != 0
    assert runtime.regions.step_pad.local_ptr != 0

    # Region sizes (bytes) agree across ranks.
    sizes = (
        runtime.regions.comm_buf.tensor.nbytes,
        runtime.regions.signal_buf.tensor.nbytes,
        runtime.regions.step_pad.tensor.nbytes,
    )
    gathered: list = [None] * runtime.world_size
    dist.all_gather_object(gathered, sizes)
    for r, s in enumerate(gathered):
        assert s == sizes, f"size mismatch: rank 0 sees {sizes}, rank {r} sees {s}"

    # Session nonce non-zero and identical across ranks.
    nonces: list = [None] * runtime.world_size
    dist.all_gather_object(nonces, runtime.session_nonce)
    assert all(n == nonces[0] for n in nonces), f"nonce mismatch: {nonces}"
    assert nonces[0] != 0

    # Group registration is idempotent and ordered.
    g_all = runtime.register_group(tuple(range(runtime.world_size)))
    g_all2 = runtime.register_group(tuple(range(runtime.world_size)))
    assert g_all is g_all2
    assert g_all.local_index == runtime.rank
    assert g_all.size == runtime.world_size

    # An ordered alias of the same membership is a *different* group.
    if runtime.world_size >= 2:
        rev = tuple(reversed(range(runtime.world_size)))
        g_rev = runtime.register_group(rev)
        assert g_rev.group_id != g_all.group_id
        assert g_rev.local_index == rev.index(runtime.rank)

    rank_print(
        f"ok: cb={cb_ptrs:#x} sb={sb_ptrs:#x} nonce={runtime.session_nonce:#x} "
        f"groups={len(runtime._registered)}"
    )

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
