"""TMA capability probe over peer pointers.

Section 7 of the design spec. Each rank produces a known byte pattern in its
own ``comm_buf[0]``, world-barriers, then constructs a 1D TMA descriptor over
its peer's ``comm_buf`` base (the peer-pointer the local rank holds) and
loads via the descriptor into a local probe buffer. If the loaded bytes match
the peer's pattern, this rank's local probe is True; otherwise False.

Results are AND-reduced across ranks via TCPStore so all ranks see the same
verdict. If ``use_tma=True`` is configured but the probe returns False the
caller raises :class:`TMAUnsupportedError`. No silent fallback.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from gfc.logging import get_logger


_ALLOCATOR_INSTALLED = False


def _install_torch_allocator() -> None:
    """Register a torch-backed Triton allocator. ``tl.make_tensor_descriptor``
    requires a workspace allocation at launch time; without an allocator the
    kernel launch raises before any memory access reaches the GPU.
    """
    global _ALLOCATOR_INSTALLED
    if _ALLOCATOR_INSTALLED:
        return

    def _alloc(size: int, alignment: int, stream):  # noqa: ARG001
        return torch.empty(size, dtype=torch.uint8, device="cuda")

    triton.set_allocator(_alloc)
    _ALLOCATOR_INSTALLED = True

if TYPE_CHECKING:
    from gfc.runtime import SymmetricCollectiveRuntime


_PROBE_BYTES = 4096
_BLOCK_BYTES = 4096  # one TMA tile spanning the probe


class TMAUnsupportedError(RuntimeError):
    """Raised when ``use_tma=True`` is requested but the probe fails."""


class TMARequirementError(ValueError):
    """Raised when a per-call TMA tile / alignment requirement is unsatisfied."""


# -----------------------------------------------------------------------------
# Probe kernel
# -----------------------------------------------------------------------------


@triton.jit
def _tma_probe_kernel(
    peer_ptr_u64,
    local_dst_u64,
    PROBE_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 1D TMA descriptor over `PROBE_BYTES` bytes at the peer base.
    peer_ptr = peer_ptr_u64.to(tl.pointer_type(tl.uint8))
    desc = tl.make_tensor_descriptor(
        peer_ptr,
        shape=[PROBE_BYTES],
        strides=[1],
        block_shape=[BLOCK],
    )
    x = tl.load_tensor_descriptor(desc, [0])
    dst_p = local_dst_u64.to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, BLOCK)
    tl.store(dst_p + offs, x, mask=offs < PROBE_BYTES)


def _pattern(rank: int, n: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.int64, device=device)
    return ((((rank + 1) * 131) + idx * 17) & 0xFF).to(torch.uint8)


def probe_tma_supported(runtime: "SymmetricCollectiveRuntime") -> bool:
    """Run the peer-pointer TMA probe and AND-reduce across world.

    Returns True iff every rank successfully read its peer's pattern via TMA.
    Caller is responsible for raising :class:`TMAUnsupportedError` if the
    user asked for TMA and the probe returned False.
    """
    log = get_logger()
    self_rank = runtime.rank
    world = runtime.world_size
    device = runtime.device
    _install_torch_allocator()

    # Step 1: each rank stages its pattern into its own comm_buf[0:PROBE_BYTES].
    pattern_self = _pattern(self_rank, _PROBE_BYTES, device)
    cb = runtime.regions.comm_buf
    cb.tensor[:_PROBE_BYTES].copy_(pattern_self)

    # Step 2: world barrier on the bootstrap PG to make the staging visible.
    dist.barrier(group=runtime.world_pg)

    # Step 3: build a descriptor for peer = (self_rank + 1) % world and load.
    peer = (self_rank + 1) % world
    peer_ptr = int(cb.handle.buffer_ptrs[peer])
    local_probe = torch.empty(_PROBE_BYTES, dtype=torch.uint8, device=device)

    try:
        with torch.cuda.stream(runtime.stream):
            _tma_probe_kernel[(1,)](
                peer_ptr,
                int(local_probe.data_ptr()),
                PROBE_BYTES=_PROBE_BYTES,
                BLOCK=_BLOCK_BYTES,
                num_warps=4,
            )
        runtime.stream.synchronize()
    except Exception as e:
        log.warning("rank %d: TMA probe kernel raised: %s", self_rank, e)
        local_ok = 0
    else:
        ref = _pattern(peer, _PROBE_BYTES, device)
        local_ok = 1 if torch.equal(local_probe, ref) else 0
        if not local_ok:
            n_mismatch = int((local_probe != ref).sum().item())
            log.warning(
                "rank %d: TMA probe read mismatched bytes from peer %d (%d/%d wrong)",
                self_rank, peer, n_mismatch, _PROBE_BYTES,
            )

    # Step 4: AND-reduce across world via dist.all_reduce.
    tally = torch.tensor([local_ok], dtype=torch.int32, device=device)
    dist.all_reduce(tally, op=dist.ReduceOp.MIN, group=runtime.world_pg)
    ok = int(tally.item()) == 1

    if self_rank == 0:
        log.info("TMA probe: %s", "ok" if ok else "FAILED")
    return ok
