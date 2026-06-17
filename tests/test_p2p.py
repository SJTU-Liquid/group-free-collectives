"""Phase-7 p2p_put + p2p_get test on 2 ranks, 100 epochs each.

The push test exercises ``p2p_put`` on the sender paired with
``p2p_put_recv`` on the receiver — the actual remote-write data path.

The pull test exercises ``p2p_get`` on the receiver paired with
``p2p_get_serve`` on the sender — the pull_copy data path.

Both: deterministic per-epoch byte pattern, auto-staging from a non-symm
source tensor allocated with ``torch.empty`` (not via symm memory).

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_p2p.py
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


def _pattern(epoch: int, role: str, nbytes: int, device) -> torch.Tensor:
    """Pattern bytes for (epoch, role)."""
    role_byte = {"put": 0xA0, "get": 0xB0}[role]
    idx = torch.arange(nbytes, dtype=torch.int64, device=device)
    p = (((epoch + 1) * 1009) + role_byte + idx * 17) & 0xFF
    return p.to(torch.uint8)


def _run_push(runtime: SymmetricCollectiveRuntime, nbytes: int, n_iter: int) -> None:
    """Sender (rank 0) -> Receiver (rank 1) using p2p_put / p2p_put_recv.

    Named ``_run_*`` (not ``test_*``) so ``pytest`` does not collect it as a
    no-fixture test: it is a torchrun-driven helper called from ``main``.
    """
    for ep in range(n_iter):
        if runtime.rank == 0:
            src = _pattern(ep, "put", nbytes, runtime.device)
            runtime.p2p_put(dst_rank=1, src=src)
        else:
            dst = torch.empty(nbytes, dtype=torch.uint8, device=runtime.device)
            runtime.p2p_put_recv(src_rank=0, dst=dst)
            runtime.stream.synchronize()
            ref = _pattern(ep, "put", nbytes, runtime.device)
            if not torch.equal(dst, ref):
                first = int((dst != ref).nonzero(as_tuple=False)[0, 0])
                raise AssertionError(
                    f"p2p_put epoch {ep}: byte {first} got=0x{int(dst[first]):02x} "
                    f"exp=0x{int(ref[first]):02x}"
                )
        # Keep sender and receiver tightly aligned in their submission order.
        dist.barrier(group=runtime.world_pg)
    runtime.stream.synchronize()


def _run_pull(runtime: SymmetricCollectiveRuntime, nbytes: int, n_iter: int) -> None:
    """Sender (rank 0) serves; Receiver (rank 1) pulls using p2p_get_serve / p2p_get.

    Named ``_run_*`` (not ``test_*``) so ``pytest`` does not collect it.
    """
    for ep in range(n_iter):
        if runtime.rank == 0:
            src = _pattern(ep, "get", nbytes, runtime.device)
            runtime.p2p_get_serve(dst_rank=1, src=src)
        else:
            dst = torch.empty(nbytes, dtype=torch.uint8, device=runtime.device)
            runtime.p2p_get(src_rank=0, dst=dst)
            runtime.stream.synchronize()
            ref = _pattern(ep, "get", nbytes, runtime.device)
            if not torch.equal(dst, ref):
                first = int((dst != ref).nonzero(as_tuple=False)[0, 0])
                raise AssertionError(
                    f"p2p_get epoch {ep}: byte {first} got=0x{int(dst[first]):02x} "
                    f"exp=0x{int(ref[first]):02x}"
                )
        dist.barrier(group=runtime.world_pg)
    runtime.stream.synchronize()


def main() -> int:
    device = setup_nccl()
    assert dist.get_world_size() == 2, "test_p2p expects nproc=2"

    config = SymmetricCollectiveConfig(
        max_group_size=2,
        max_collective_bytes=1 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    for nbytes in (256, 4096, 64 * 1024):
        _run_push(runtime, nbytes, n_iter=100)
        _run_pull(runtime, nbytes, n_iter=100)

    rank_print("ok p2p_put + p2p_get, 100 epochs each, sizes 256/4 KiB/64 KiB")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
