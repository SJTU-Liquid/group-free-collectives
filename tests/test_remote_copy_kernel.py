"""Phase-3 test: standalone ``pull_copy_kernel`` between 2 ranks.

Layout: rank 0 writes a deterministic byte pattern into its own ``comm_buf[0]``,
world-barrier on the bootstrap PG (so the write is globally visible — note that
between *distinct* kernel launches on the same device the release/acquire-sys
barrier kernel would be required, but here we use the bootstrap barrier
because we are explicitly testing the byte-copy kernel in isolation), then
rank 1 issues ``pull_copy`` from rank 0's symmetric base into a local
verification tensor and checks bytes match.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_remote_copy_kernel.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import SymmetricCollectiveConfig, SymmetricCollectiveRuntime  # noqa: E402
from gfc.kernels._common import vec_width_bytes  # noqa: E402
from gfc.kernels.copy_pull import launch_pull_copy  # noqa: E402

from tests._harness import setup_nccl, teardown, rank_print  # noqa: E402


def _pattern(nbytes: int, seed: int) -> torch.Tensor:
    """Reproducible CPU pattern. ``pattern[i] = ((seed + 1) * 131 + i * 17) & 0xFF``."""
    idx = torch.arange(nbytes, dtype=torch.int64)
    p = (((seed + 1) * 131) + idx * 17) & 0xFF
    return p.to(torch.uint8)


def _check_one(
    runtime: SymmetricCollectiveRuntime,
    nbytes: int,
    slice_offset: int,
    force_vec: int | None = None,
) -> None:
    """Rank 0 stages pattern into its comm_buf, rank 1 pulls and verifies."""
    cb = runtime.regions.comm_buf
    handle = cb.handle
    if runtime.rank == 0:
        pat = _pattern(nbytes, seed=runtime.session_nonce & 0xFF).to(runtime.device)
        local_view = handle.get_buffer(rank=0, sizes=(nbytes,), dtype=torch.uint8, storage_offset=slice_offset)
        local_view.copy_(pat)
    # Ensure the staging copy is visible globally before rank 1 pulls.
    # Bootstrap barrier is allowed in tests.
    dist.barrier(group=runtime.world_pg)

    if runtime.rank == 1:
        dst = torch.empty(nbytes, dtype=torch.uint8, device=runtime.device)

        # Peer base address (rank 0's comm_buf base on this rank's address space)
        peer_ptrs = handle.buffer_ptrs  # python list of ints, one per peer
        src_ptr = int(peer_ptrs[0]) + slice_offset
        dst_ptr = int(dst.data_ptr())

        align = min(
            src_ptr & -src_ptr if src_ptr else 1 << 30,
            dst_ptr & -dst_ptr if dst_ptr else 1 << 30,
        )
        vec = force_vec if force_vec is not None else vec_width_bytes(align, align, nbytes)

        with torch.cuda.stream(runtime.stream):
            launch_pull_copy(
                src_ptr=src_ptr,
                dst_ptr=dst_ptr,
                nbytes=nbytes,
                vec_bytes=vec,
            )
        runtime.stream.synchronize()

        ref = _pattern(nbytes, seed=runtime.session_nonce & 0xFF)
        got = dst.cpu()
        mismatches = (got != ref).nonzero(as_tuple=False)
        if mismatches.numel() > 0:
            first = int(mismatches[0, 0])
            raise AssertionError(
                f"pull_copy mismatch at byte {first}/{nbytes} "
                f"(slice_offset={slice_offset}, vec={vec}): "
                f"got=0x{int(got[first]):02x} ref=0x{int(ref[first]):02x}"
            )
        rank_print(
            f"ok nbytes={nbytes} offset={slice_offset} vec={vec} "
            f"first_byte=0x{int(got[0]):02x}"
        )

    # Re-sync before reusing comm_buf for the next case.
    dist.barrier(group=runtime.world_pg)


def main() -> int:
    device = setup_nccl()
    assert dist.get_world_size() == 2, "test_remote_copy_kernel expects nproc=2"

    config = SymmetricCollectiveConfig(
        max_group_size=2,
        max_collective_bytes=4 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    # Cover several interesting cases: fully aligned vec=16, vec=8 only,
    # vec=4 only, and a tail-bytes case.
    cases = [
        (1 * 1024 * 1024, 0, None),   # 1 MiB, aligned -> auto vec=16
        (256 * 1024, 16, None),       # offset 16 keeps 16-alignment
        (12_345, 0, None),            # auto chooses vec=1
        (5, 0, None),                 # tiny payload
        # Explicit vec to exercise the HAS_TAIL path: vec=8 with 5-byte tail.
        (16 * 1024 + 5, 0, 8),
        # Explicit vec=4 with 3-byte tail.
        (4096 + 3, 0, 4),
    ]
    for nbytes, off, force_vec in cases:
        _check_one(runtime, nbytes, off, force_vec)

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
