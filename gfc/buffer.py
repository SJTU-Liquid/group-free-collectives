"""Symmetric-memory region allocation.

Section 1 of the spec. Three regions are allocated once at runtime init and
never reallocated:

* ``comm_buf``      : ``uint8 [max_collective_bytes]`` — the data plane.
* ``signal_buf``    : ``uint64 [2 slots, world_size src]`` — per-edge
  double-buffered arrive token grid. Cell ``[s, p]`` on rank q is written
  remotely by rank p when its local edge sequence with q has parity ``s``,
  and read locally by rank q with acquire semantics. The two slots rotate
  per pair, so consecutive barriers between the same pair never collide
  on the same cell. The protocol has no separate finish/ack phase — slot
  reuse is delayed by one barrier, and pairwise-ordered stream-FIFO
  guarantees the peer has consumed the previous value before reuse.
* ``step_pad``      : ``uint64 [fused channels, chunks, world_size]`` — fused path steps.

There is no host-side pointer table passed to kernels; each kernel reads the
device-resident ``buffer_ptrs_dev()`` ``void**`` directly off the symmetric
handle.

The per-rank ``edge_seq`` counter (uint64[world_size]) that drives the slot
selection is a *local* tensor — peers do not read it — so it lives on the
runtime, not in this symmetric-regions container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed._symmetric_memory import empty as sm_empty
from torch.distributed._symmetric_memory import rendezvous as sm_rendezvous

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed._symmetric_memory import _SymmetricMemory

    from gfc.config import SymmetricCollectiveConfig


@dataclass
class SymmRegion:
    """A single allocated + rendezvoused symmetric region."""

    tensor: "torch.Tensor"
    handle: "_SymmetricMemory"

    @property
    def buffer_ptrs_dev(self) -> int:
        return int(self.handle.buffer_ptrs_dev)

    @property
    def local_ptr(self) -> int:
        return int(self.tensor.data_ptr())


@dataclass
class SymmRegions:
    comm_buf: SymmRegion
    signal_buf: SymmRegion
    # Per-(channel, chunk, producer) signal cell used by the fused single-
    # kernel data path. Each rank holds its own
    # ``uint64[num_channels, max_chunks_per_channel, world_size]`` view.
    # Cell ``[ch, k, p]`` on rank q is written remotely by rank p (the
    # channel-c sender CTA, completing chunk k) with ``st.release.sys`` and
    # read locally by rank q's receiver CTA with ``ld.acquire.sys``. Tokens
    # are unique per-(collective, chunk) so a stale cell from a prior
    # collective never satisfies the next collective's wait.
    step_pad: SymmRegion


def allocate_symm_regions(
    config: "SymmetricCollectiveConfig",
    device: "torch.device",
    world_pg: "ProcessGroup",
    world_size: int,
) -> SymmRegions:
    """Allocate + rendezvous the three regions on the world process group.

    All three are rendezvoused on ``world_pg``. Subgroups are *not* given their
    own process groups; subgroup collectives use the world-rendezvoused peer
    pointer tables directly with rank-set masking inside kernels.
    """
    comm_t = sm_empty(
        config.max_collective_bytes,
        dtype=torch.uint8,
        device=device,
    )
    signal_t = sm_empty(
        config.num_signal_slots * world_size,
        dtype=torch.uint64,
        device=device,
    )
    step_pad_t = sm_empty(
        config.fused_num_channels
        * config.fused_max_chunks_per_channel
        * world_size,
        dtype=torch.uint64,
        device=device,
    )

    # Zero locally so a fresh boot starts clean; rendezvous itself does not
    # zero the backing pages. Kernels later store with full overwrite, so
    # this only matters for the first acquire-spin to read a non-token.
    comm_t.zero_()
    signal_t.zero_()
    step_pad_t.zero_()

    # Rendezvous each region on the world group.
    comm_h = sm_rendezvous(comm_t, world_pg)
    signal_h = sm_rendezvous(signal_t, world_pg)
    step_pad_h = sm_rendezvous(step_pad_t, world_pg)

    return SymmRegions(
        comm_buf=SymmRegion(tensor=comm_t, handle=comm_h),
        signal_buf=SymmRegion(tensor=signal_t, handle=signal_h),
        step_pad=SymmRegion(tensor=step_pad_t, handle=step_pad_h),
    )


def zero_signal_grids(regions: SymmRegions) -> None:
    """Zero the signal / step grids on the local rank's symmetric tensor.

    Used between tests or at shutdown. Call must be world-synchronized at the
    boundary (e.g., via ``handle.barrier()`` or bootstrap barrier).
    """
    regions.signal_buf.tensor.zero_()
    regions.step_pad.tensor.zero_()
