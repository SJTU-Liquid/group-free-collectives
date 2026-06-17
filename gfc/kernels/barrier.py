"""Release/acquire-sys barrier over a subgroup, per-edge double-buffered.

The signal grid is laid out per rank as ``uint64[2 slots, world_size src]``,
i.e. for each ordered pair ``(src -> dst)`` there are two cells. Each rank
keeps a *local* ``edge_seq[world_size]`` counter; barrier ``N`` on the edge
``(self, peer)`` uses ``slot = edge_seq[peer] & 1`` and then bumps the
counter. Because both endpoints of the pair process the same sequence of
pair-involving barriers (the scheduler's "pairwise order consistent"
invariant), the two ranks compute matching slots without ever exchanging
slot state.

The protocol has no separate finish/ack phase. Each rank publishes its
token to the peer's cell once and waits for the peer's reciprocal cell —
slot reuse is delayed by one barrier, and by the time rank r would
overwrite slot S (i.e., on barrier N+2), the prior barrier N+1 has already
returned, which means peer has past barrier N+1's publish, which means
peer has past barrier N's wait, which means peer already consumed slot S.
So the double buffer is sufficient on its own.

Memory ordering:
  * ``st.global.release.sys.b64`` makes every prior store visible to any
    observer that subsequently reads the same address with acquire
    semantics at ``sys`` scope. This carries through prior kernels on the
    same CUDA stream.
  * ``ld.global.acquire.sys.b64`` is enough for the wait side because each
    slot has exactly one writer for the current token.
"""

from __future__ import annotations

import triton
import triton.language as tl

from gfc.kernels._ptx import (
    ld_acquire_sys_u64,
    read_globaltimer,
    st_release_sys_u64,
    trap_if,
)


@triton.jit
def barrier_kernel(
    token,                              # tl.uint64
    self_global_rank,                   # tl.int32
    group_ranks_row_u64,                # uint64-packed addr of uint32[max_group_size] row
    signal_ptrs_dev_u64,                # uint64-packed addr of uint64[world_size] table
    edge_seq_ptr_u64,                   # uint64 addr of LOCAL uint64[world_size] counter
    WORLD_SIZE: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,           # 0 disables the watchdog
):
    """Per-peer barrier. Launch grid is ``(group.size,)``; each program is a
    single-lane scalar program that handles one (self, peer) edge."""
    pid = tl.program_id(0)

    # Ordered device-resident rank list for this group. Each entry is uint32.
    group_ranks_ptr = group_ranks_row_u64.to(tl.pointer_type(tl.uint32))
    peer = tl.load(group_ranks_ptr + pid)              # uint32 scalar

    # Peer-pointer table — entry r is the uint64 address of rank r's signal_buf
    # base in this rank's address space (NVLink P2P virtual address).
    sig_ptrs = signal_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))
    peer_base = tl.load(sig_ptrs + peer)
    self_base = tl.load(sig_ptrs + self_global_rank)

    # Read and bump the LOCAL edge_seq[peer] counter to derive this barrier's
    # double-buffer slot for the (self, peer) edge. All barriers run on the
    # runtime's single submission stream, so stream-FIFO serializes successive
    # barriers — plain load/store is sufficient; no atomic needed.
    #
    # CUDA Graph replay is not a supported runtime mode in v1. The counter is
    # GPU-resident, but the host-computed token is captured as a launch arg.
    edge_seq_ptr = edge_seq_ptr_u64.to(tl.pointer_type(tl.uint64))
    seq_cell = edge_seq_ptr + peer
    seq = tl.load(seq_cell)
    slot = seq & 1
    tl.store(seq_cell, seq + 1)

    # ---- publish: write peer's incoming arrive cell ------------------------
    # Layout per rank: uint64[2 slots, WORLD_SIZE src].
    # Address of "peer's incoming cell from self at slot s" within peer's
    # signal_buf: ``s * WORLD_SIZE + self_global_rank`` u64 entries.
    write_off = slot * WORLD_SIZE + self_global_rank
    write_ptr = (peer_base + write_off * 8).to(tl.pointer_type(tl.uint64))
    st_release_sys_u64(write_ptr, token)

    # ---- wait: peer's symmetric publish into my signal_buf -----------------
    read_off = slot * WORLD_SIZE + peer
    read_ptr = (self_base + read_off * 8).to(tl.pointer_type(tl.uint64))

    # Acquire spin. When a timeout is configured, bound the wait with the
    # device wall clock and ``trap`` on expiry, converting an otherwise-
    # unbounded hang (mismatched barrier order, dead peer, lost token) into a
    # diagnosable CUDA error on the host's next sync.
    if TIMEOUT_NS > 0:
        deadline = read_globaltimer() + TIMEOUT_NS
    observed = ld_acquire_sys_u64(read_ptr)
    while observed != token:
        observed = ld_acquire_sys_u64(read_ptr)
        if TIMEOUT_NS > 0:
            trap_if(read_globaltimer() > deadline)


# -----------------------------------------------------------------------------
# Host launcher
# -----------------------------------------------------------------------------


def launch_barrier(
    *,
    token: int,
    self_global_rank: int,
    group_ranks_row_ptr: int,
    signal_ptrs_dev: int,
    edge_seq_ptr: int,
    group_size: int,
    world_size: int,
    timeout_ns: int = 0,
) -> None:
    """Launch the barrier kernel on the current CUDA stream.

    Caller must enter the runtime stream context first. ``timeout_ns`` (0 to
    disable) arms an in-kernel watchdog that traps if the acquire spin exceeds
    the bound, so a hung barrier fails loudly instead of spinning forever.
    """
    assert group_size > 0
    grid = (int(group_size),)
    barrier_kernel[grid](
        int(token),
        int(self_global_rank),
        int(group_ranks_row_ptr),
        int(signal_ptrs_dev),
        int(edge_seq_ptr),
        WORLD_SIZE=int(world_size),
        TIMEOUT_NS=int(timeout_ns),
        num_warps=1,
        num_stages=1,
    )
