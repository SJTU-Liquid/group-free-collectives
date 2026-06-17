"""Fused single-kernel data path for all_gather / all2all.

Replaces the legacy ``stage -> barrier kernel -> pull kernel -> barrier kernel``
sequence with a single persistent kernel:

* Grid is ``2 * num_channels`` CTAs. CTA ``2c`` (the "sender" CTA) and
  CTA ``2c+1`` (the "receiver" CTA) form a pair owning channel ``c``'s
  partition of the input/output slice.
* Sender CTA iterates over chunks within its partition. For each chunk K:
    1. Stage own input chunk into own ``comm_buf`` at the chunk's offset.
    2. Self-copy: also write the same chunk to ``output[self_position +
       partition_off + chunk_off]`` so receivers can skip the self peer.
    3. ``fence.acq_rel.sys`` to make the stage globally visible.
    4. Publish ``step_base + K + 1`` to every peer's ``step_pad`` cell at
       ``[channel][self_global_rank]`` via ``st.release.sys.b64``.
* Receiver CTA iterates over chunks within its partition. For each chunk K:
    1. Spin on ``ld.acquire.sys.b64`` of own ``step_pad[channel][peer]`` for
       every peer in the group, waiting for ``>= step_base + K + 1``.
    2. Pull the chunk from each peer's ``comm_buf`` into the local output at
       ``output[peer_position + partition_off + chunk_off]``.

Alignment: per-peer slice bases are at ``peer_idx * slice_bytes``; when
``slice_bytes`` is not 16-byte aligned, the slice starts themselves are
misaligned and a hard ``ld.global.v4.b32`` would fault. The kernel takes
``VEC_BYTES`` (1/4/8/16) as a constexpr; the host picks
``gcd(16, slice_bytes_align, input_ptr_align, output_ptr_align)`` so the
fast path runs whenever possible and degrades gracefully (down to one
byte per lane) when callers hand in awkward sizes. Production workloads
(bf16, hidden-dim multiples) hit ``VEC_BYTES == 16`` every time; the
narrower variants exist so the dispatch never needs to fall back to a
separate kernel.

The collective entry runs one token pre-barrier before this kernel and one
token post-barrier after it (both via the legacy ``launch_barrier``). The
in-kernel step counters order each receiver's pulls against the matching
sender's stage, but they only prove *this* rank finished its own pulls — not
that peers finished pulling from this rank's ``comm_buf``. The host
post-barrier supplies that cross-rank acknowledgement, so a following
collective (possibly on a different, overlapping group) cannot overwrite
``comm_buf`` while a peer is still reading it.
"""

from __future__ import annotations

import triton
import triton.language as tl

from gfc.kernels._common import base_alignment, vec_width_bytes
from gfc.kernels._ptx import (
    fence_acq_rel_sys,
    ld_acquire_sys_u64,
    ld_global_v4_b32_cond,
    read_globaltimer,
    st_global_v4_b32_cond,
    st_release_sys_u64_pred,
    trap_if,
)


_DEFAULT_NUM_WARPS = 4               # CTA threads = 128
_DEFAULT_BLOCK_ELTS = 4096           # tile lanes; per-iter bytes = BLOCK_ELTS * VEC_BYTES


def _next_pow2(n: int) -> int:
    """Smallest power-of-2 >= n. Used to round the per-lane group dimension
    up so ``tl.arange`` (which requires a power-of-2 size) can host arbitrary
    subgroup sizes; lanes ``>= group_size`` are masked out."""
    assert n >= 1
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _byte_tail_copy(
    src_base_u64,
    dst1_base_u64,
    dst2_base_u64,
    off,                           # int64: byte offset into the segment
    nbytes_total,                  # int64: total bytes in the segment
    HAS_DST2: tl.constexpr,
    BLOCK_ELTS: tl.constexpr,
):
    """Byte-wise copy of ``[off, nbytes_total)``. All CTA threads cooperate;
    only the lanes whose absolute byte index is in range write."""
    lane = tl.arange(0, BLOCK_ELTS).to(tl.int64)
    for inner in tl.range(off, nbytes_total, BLOCK_ELTS):
        idx = inner + lane
        mask = idx < nbytes_total
        sp = src_base_u64.to(tl.pointer_type(tl.uint8))
        x = tl.load(sp + idx, mask=mask)
        tl.store(dst1_base_u64.to(tl.pointer_type(tl.uint8)) + idx, x, mask=mask)
        if HAS_DST2:
            tl.store(
                dst2_base_u64.to(tl.pointer_type(tl.uint8)) + idx,
                x,
                mask=mask,
            )


@triton.jit
def _fused_block_copy(
    src_base_u64,
    dst1_base_u64,
    dst2_base_u64,
    nbytes,                          # int64: bytes to copy
    HAS_DST2: tl.constexpr,
    BLOCK_ELTS: tl.constexpr,
    VEC_BYTES: tl.constexpr,
):
    """Tile-parallel copy of ``nbytes`` from ``src_base`` to ``dst1_base``
    (and optionally ``dst2_base``) at ``VEC_BYTES`` per lane.

    Bulk: each iteration copies ``BLOCK_ELTS * VEC_BYTES`` bytes via
    vectorized load/store. ``nbytes`` does NOT need to be a multiple of
    ``VEC_BYTES`` — the trailing ``nbytes % VEC_BYTES`` bytes are handled
    by a byte-wise tail using ``_byte_tail_copy``, so callers don't have
    to ensure exact-multiple sizes.

    The bases themselves must satisfy the active vec width's alignment
    (the host picks ``VEC_BYTES`` so this holds at call time).
    """
    if VEC_BYTES == 1:
        # Pure byte path — no bulk/tail split needed.
        _byte_tail_copy(
            src_base_u64,
            dst1_base_u64,
            dst2_base_u64,
            0,
            nbytes,
            HAS_DST2=HAS_DST2,
            BLOCK_ELTS=BLOCK_ELTS,
        )
        return

    # Bulk: covers ``nbytes // VEC_BYTES * VEC_BYTES`` bytes.
    bulk_bytes = (nbytes // VEC_BYTES) * VEC_BYTES
    block_bytes = BLOCK_ELTS * VEC_BYTES
    lane = tl.arange(0, BLOCK_ELTS).to(tl.int64)
    for off in tl.range(0, bulk_bytes, block_bytes):
        remaining = bulk_bytes - off
        if VEC_BYTES == 16:
            byte_offs = lane * 16
            mask = byte_offs < remaining
            sp = (src_base_u64 + off).to(tl.pointer_type(tl.uint8)) + byte_offs
            v0, v1, v2, v3 = ld_global_v4_b32_cond(sp, mask)
            d1 = (dst1_base_u64 + off).to(tl.pointer_type(tl.uint8)) + byte_offs
            st_global_v4_b32_cond(d1, v0, v1, v2, v3, mask)
            if HAS_DST2:
                d2 = (dst2_base_u64 + off).to(tl.pointer_type(tl.uint8)) + byte_offs
                st_global_v4_b32_cond(d2, v0, v1, v2, v3, mask)
        elif VEC_BYTES == 8:
            mask = (lane * 8) < remaining
            sp = (src_base_u64 + off).to(tl.pointer_type(tl.uint64))
            x = tl.load(sp + lane, mask=mask)
            tl.store((dst1_base_u64 + off).to(tl.pointer_type(tl.uint64)) + lane, x, mask=mask)
            if HAS_DST2:
                tl.store((dst2_base_u64 + off).to(tl.pointer_type(tl.uint64)) + lane, x, mask=mask)
        else:  # VEC_BYTES == 4
            mask = (lane * 4) < remaining
            sp = (src_base_u64 + off).to(tl.pointer_type(tl.uint32))
            x = tl.load(sp + lane, mask=mask)
            tl.store((dst1_base_u64 + off).to(tl.pointer_type(tl.uint32)) + lane, x, mask=mask)
            if HAS_DST2:
                tl.store((dst2_base_u64 + off).to(tl.pointer_type(tl.uint32)) + lane, x, mask=mask)

    # Tail: ``nbytes - bulk_bytes`` bytes (0..VEC_BYTES-1) handled byte-wise.
    if bulk_bytes < nbytes:
        _byte_tail_copy(
            src_base_u64,
            dst1_base_u64,
            dst2_base_u64,
            bulk_bytes,
            nbytes,
            HAS_DST2=HAS_DST2,
            BLOCK_ELTS=BLOCK_ELTS,
        )


@triton.jit
def all_gather_fused_kernel(
    input_u64,                       # uint64-packed: local input base
    output_u64,                      # uint64-packed: local output base
    comm_buf_ptrs_dev_u64,           # addr of uint64[world_size] peer comm_buf bases
    step_pad_ptrs_dev_u64,           # addr of uint64[world_size] peer step_pad bases
    group_ranks_row_u64,             # addr of uint32[max_group_size] ordered ranks
    comm_buf_offset,                # int64
    self_global_rank,                # int32
    self_local_index,                # int32
    slice_bytes,                     # int64: per-peer slice
    partition_bytes,                 # int64: aligned partition per channel
    chunk_bytes,                     # int64: chunk size within partition
    step_base,                       # uint64
    WORLD_SIZE: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_LANES: tl.constexpr,
    NUM_CHANNELS: tl.constexpr,
    BLOCK_ELTS: tl.constexpr,
    VEC_BYTES: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,
):
    pid = tl.program_id(0)
    channel = pid // 2
    is_sender = (pid % 2) == 0

    partition_off = channel.to(tl.int64) * partition_bytes
    if partition_off >= slice_bytes:
        return
    partition_size = tl.minimum(partition_bytes, slice_bytes - partition_off)
    n_chunks = tl.cdiv(partition_size, chunk_bytes)

    group_ranks_ptr = group_ranks_row_u64.to(tl.pointer_type(tl.uint32))
    cb_peer_ptrs = comm_buf_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))
    sp_peer_ptrs = step_pad_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))

    own_comm_buf_base = tl.load(cb_peer_ptrs + self_global_rank)
    own_step_pad_base = tl.load(sp_peer_ptrs + self_global_rank)

    self_off = tl.cast(self_local_index, tl.int64) * slice_bytes

    if is_sender:
        for k in tl.range(n_chunks):
            chunk_off = k.to(tl.int64) * chunk_bytes
            this_chunk = tl.minimum(chunk_bytes, partition_size - chunk_off)
            base_byte = partition_off + chunk_off
            _fused_block_copy(
                input_u64 + base_byte,
                own_comm_buf_base + comm_buf_offset + base_byte,
                output_u64 + self_off + base_byte,
                this_chunk,
                HAS_DST2=True,
                BLOCK_ELTS=BLOCK_ELTS,
                VEC_BYTES=VEC_BYTES,
            )

            tl.debug_barrier()
            _ = fence_acq_rel_sys()

            peer_lane = tl.arange(0, GROUP_LANES)
            in_group = peer_lane < GROUP_SIZE
            peer = tl.load(
                group_ranks_ptr + peer_lane, mask=in_group, other=0
            ).to(tl.int64)
            peer_step_pad_base = tl.load(
                sp_peer_ptrs + peer, mask=in_group, other=0
            )
            cell_off = (
                channel.to(tl.int64) * MAX_CHUNKS * WORLD_SIZE
                + k.to(tl.int64) * WORLD_SIZE
                + tl.cast(self_global_rank, tl.int64)
            ) * 8
            write_ptr = (peer_step_pad_base + cell_off).to(tl.pointer_type(tl.uint64))
            do_write = in_group & (peer_lane != self_local_index)
            pub_val = tl.cast(step_base, tl.uint64) + (k.to(tl.uint64) + 1)
            _ = st_release_sys_u64_pred(write_ptr, pub_val, do_write.to(tl.int32))
    else:
        for k in tl.range(n_chunks):
            chunk_off = k.to(tl.int64) * chunk_bytes
            this_chunk = tl.minimum(chunk_bytes, partition_size - chunk_off)
            target = tl.cast(step_base, tl.uint64) + (k.to(tl.uint64) + 1)

            peer_lane = tl.arange(0, GROUP_LANES)
            in_group = peer_lane < GROUP_SIZE
            peer = tl.load(
                group_ranks_ptr + peer_lane, mask=in_group, other=0
            ).to(tl.int64)
            cell_off = (
                channel.to(tl.int64) * MAX_CHUNKS * WORLD_SIZE
                + k.to(tl.int64) * WORLD_SIZE
                + peer
            ) * 8
            read_ptr = (own_step_pad_base + cell_off).to(tl.pointer_type(tl.uint64))
            is_self_or_unused = (peer_lane == self_local_index) | (~in_group)

            if TIMEOUT_NS > 0:
                deadline = read_globaltimer() + TIMEOUT_NS
            observed = ld_acquire_sys_u64(read_ptr)
            satisfied = is_self_or_unused | (observed == target)
            all_done = tl.min(satisfied.to(tl.int32), axis=0)
            while all_done == 0:
                observed = ld_acquire_sys_u64(read_ptr)
                satisfied = is_self_or_unused | (observed == target)
                all_done = tl.min(satisfied.to(tl.int32), axis=0)
                if TIMEOUT_NS > 0:
                    trap_if(read_globaltimer() > deadline)

            tl.debug_barrier()
            _ = fence_acq_rel_sys()

            base_byte = partition_off + chunk_off
            for p_delta in tl.range(GROUP_SIZE):
                peer_idx = (p_delta + self_local_index + 1) % GROUP_SIZE
                if peer_idx != self_local_index:
                    peer_rank = tl.load(group_ranks_ptr + peer_idx).to(tl.int64)
                    peer_cb_base = tl.load(cb_peer_ptrs + peer_rank)
                    peer_out_off = tl.cast(peer_idx, tl.int64) * slice_bytes
                    _fused_block_copy(
                        peer_cb_base + comm_buf_offset + base_byte,
                        output_u64 + peer_out_off + base_byte,
                        0,
                        this_chunk,
                        HAS_DST2=False,
                        BLOCK_ELTS=BLOCK_ELTS,
                        VEC_BYTES=VEC_BYTES,
                    )


@triton.jit
def all2all_fused_kernel(
    input_u64,
    output_u64,
    comm_buf_ptrs_dev_u64,
    step_pad_ptrs_dev_u64,
    group_ranks_row_u64,
    comm_buf_offset,
    self_global_rank,
    self_local_index,
    slice_bytes,
    partition_bytes,
    chunk_bytes,
    step_base,
    WORLD_SIZE: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_LANES: tl.constexpr,
    NUM_CHANNELS: tl.constexpr,
    BLOCK_ELTS: tl.constexpr,
    VEC_BYTES: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,
):
    """Fused all2all. See module docstring for the protocol.

    Layout assumptions:
      * ``input`` is laid out as ``GROUP_SIZE`` per-peer slices of
        ``slice_bytes`` bytes; slice ``t`` is destined for peer
        ``group.ranks[t]``.
      * Sender stages, per chunk: for every ``t``, copy
        ``input[t*slice_bytes + partition_off + chunk_off]`` to
        ``own_comm_buf[t*slice_bytes + partition_off + chunk_off]``;
        for ``t == self_local_index`` also write directly to ``output``.
      * Receiver pulls, per chunk: for every peer ``p != self``,
        ``peer_comm_buf[self_local_index*slice_bytes + ...]`` to
        ``output[p_idx*slice_bytes + ...]``.
    """
    pid = tl.program_id(0)
    channel = pid // 2
    is_sender = (pid % 2) == 0

    partition_off = channel.to(tl.int64) * partition_bytes
    if partition_off >= slice_bytes:
        return
    partition_size = tl.minimum(partition_bytes, slice_bytes - partition_off)
    n_chunks = tl.cdiv(partition_size, chunk_bytes)

    group_ranks_ptr = group_ranks_row_u64.to(tl.pointer_type(tl.uint32))
    cb_peer_ptrs = comm_buf_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))
    sp_peer_ptrs = step_pad_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))

    own_comm_buf_base = tl.load(cb_peer_ptrs + self_global_rank)
    own_step_pad_base = tl.load(sp_peer_ptrs + self_global_rank)

    self_slice_off = tl.cast(self_local_index, tl.int64) * slice_bytes

    if is_sender:
        for k in tl.range(n_chunks):
            chunk_off = k.to(tl.int64) * chunk_bytes
            this_chunk = tl.minimum(chunk_bytes, partition_size - chunk_off)
            base_byte = partition_off + chunk_off

            # Stage every per-peer slice into the local comm_buf.
            for t in tl.range(GROUP_SIZE):
                t_slice_off = tl.cast(t, tl.int64) * slice_bytes
                _fused_block_copy(
                    input_u64 + t_slice_off + base_byte,
                    own_comm_buf_base + comm_buf_offset + t_slice_off + base_byte,
                    0,
                    this_chunk,
                    HAS_DST2=False,
                    BLOCK_ELTS=BLOCK_ELTS,
                    VEC_BYTES=VEC_BYTES,
                )
            # Self-copy: own per-peer slice for ``self_local_index`` goes
            # straight to the local output so the receiver CTA can skip
            # the self peer entirely.
            _fused_block_copy(
                input_u64 + self_slice_off + base_byte,
                output_u64 + self_slice_off + base_byte,
                0,
                this_chunk,
                HAS_DST2=False,
                BLOCK_ELTS=BLOCK_ELTS,
                VEC_BYTES=VEC_BYTES,
            )

            tl.debug_barrier()
            _ = fence_acq_rel_sys()

            peer_lane = tl.arange(0, GROUP_LANES)
            in_group = peer_lane < GROUP_SIZE
            peer = tl.load(
                group_ranks_ptr + peer_lane, mask=in_group, other=0
            ).to(tl.int64)
            peer_step_pad_base = tl.load(
                sp_peer_ptrs + peer, mask=in_group, other=0
            )
            cell_off = (
                channel.to(tl.int64) * MAX_CHUNKS * WORLD_SIZE
                + k.to(tl.int64) * WORLD_SIZE
                + tl.cast(self_global_rank, tl.int64)
            ) * 8
            write_ptr = (peer_step_pad_base + cell_off).to(tl.pointer_type(tl.uint64))
            do_write = in_group & (peer_lane != self_local_index)
            pub_val = tl.cast(step_base, tl.uint64) + (k.to(tl.uint64) + 1)
            _ = st_release_sys_u64_pred(write_ptr, pub_val, do_write.to(tl.int32))
    else:
        for k in tl.range(n_chunks):
            chunk_off = k.to(tl.int64) * chunk_bytes
            this_chunk = tl.minimum(chunk_bytes, partition_size - chunk_off)
            target = tl.cast(step_base, tl.uint64) + (k.to(tl.uint64) + 1)

            peer_lane = tl.arange(0, GROUP_LANES)
            in_group = peer_lane < GROUP_SIZE
            peer = tl.load(
                group_ranks_ptr + peer_lane, mask=in_group, other=0
            ).to(tl.int64)
            cell_off = (
                channel.to(tl.int64) * MAX_CHUNKS * WORLD_SIZE
                + k.to(tl.int64) * WORLD_SIZE
                + peer
            ) * 8
            read_ptr = (own_step_pad_base + cell_off).to(tl.pointer_type(tl.uint64))
            is_self_or_unused = (peer_lane == self_local_index) | (~in_group)

            if TIMEOUT_NS > 0:
                deadline = read_globaltimer() + TIMEOUT_NS
            observed = ld_acquire_sys_u64(read_ptr)
            satisfied = is_self_or_unused | (observed == target)
            all_done = tl.min(satisfied.to(tl.int32), axis=0)
            while all_done == 0:
                observed = ld_acquire_sys_u64(read_ptr)
                satisfied = is_self_or_unused | (observed == target)
                all_done = tl.min(satisfied.to(tl.int32), axis=0)
                if TIMEOUT_NS > 0:
                    trap_if(read_globaltimer() > deadline)

            tl.debug_barrier()
            _ = fence_acq_rel_sys()

            base_byte = partition_off + chunk_off
            for p_delta in tl.range(GROUP_SIZE):
                peer_idx = (p_delta + self_local_index + 1) % GROUP_SIZE
                if peer_idx != self_local_index:
                    peer_rank = tl.load(group_ranks_ptr + peer_idx).to(tl.int64)
                    peer_cb_base = tl.load(cb_peer_ptrs + peer_rank)
                    peer_out_off = tl.cast(peer_idx, tl.int64) * slice_bytes
                    _fused_block_copy(
                        peer_cb_base + comm_buf_offset + self_slice_off + base_byte,
                        output_u64 + peer_out_off + base_byte,
                        0,
                        this_chunk,
                        HAS_DST2=False,
                        BLOCK_ELTS=BLOCK_ELTS,
                        VEC_BYTES=VEC_BYTES,
                    )


# -----------------------------------------------------------------------------
# Host launchers
# -----------------------------------------------------------------------------


def _pick_vec_bytes(input_ptr: int, output_ptr: int, slice_bytes: int) -> int:
    """Pick the widest legal vector width for the call.

    Considers the alignment of input/output base pointers and the per-peer
    slice stride (since peer ``t``'s slice starts at ``t * slice_bytes``).
    partition_bytes is config.alignment-aligned (>=16) by construction, so
    it doesn't constrain the choice. Returns one of ``{16, 8, 4, 1}``.
    """
    return vec_width_bytes(
        base_alignment(input_ptr),
        base_alignment(output_ptr),
        base_alignment(slice_bytes),
    )


def launch_all_gather_fused(
    *,
    input_ptr: int,
    output_ptr: int,
    comm_buf_ptrs_dev: int,
    step_pad_ptrs_dev: int,
    group_ranks_row_ptr: int,
    comm_buf_offset: int,
    self_global_rank: int,
    self_local_index: int,
    slice_bytes: int,
    group_size: int,
    world_size: int,
    max_chunks_per_channel: int,
    num_channels: int,
    chunk_bytes: int,
    step_base: int,
    alignment: int = 16,
    timeout_ns: int = 0,
) -> None:
    """Host-side launch of the fused all_gather kernel.

    The vec width is chosen automatically from the alignment of the input
    pointer, the output pointer and ``slice_bytes``. Production tensors
    (bf16 contiguous, hidden-dim multiples of 128) hit the 16-byte fast
    path; awkward sizes (e.g. ``slice_bytes`` odd) gracefully degrade.
    """
    assert slice_bytes > 0
    assert group_size > 0
    assert num_channels > 0
    assert chunk_bytes > 0

    partition_bytes = (slice_bytes + num_channels - 1) // num_channels
    partition_bytes = (partition_bytes + alignment - 1) & ~(alignment - 1)

    n_chunks = (partition_bytes + chunk_bytes - 1) // chunk_bytes
    if n_chunks > max_chunks_per_channel:
        # Hard check (not ``assert`` — must survive ``python -O``): exceeding
        # the static bound would index ``step_pad``'s chunk axis past its
        # allocation, an out-of-bounds write into symmetric memory.
        raise ValueError(
            f"per-channel chunk count {n_chunks} exceeds static bound "
            f"{max_chunks_per_channel}; raise fused_max_chunks_per_channel or "
            f"fused_chunk_size"
        )

    vec_bytes = _pick_vec_bytes(input_ptr, output_ptr, slice_bytes)

    grid = (2 * num_channels,)
    all_gather_fused_kernel[grid](
        int(input_ptr),
        int(output_ptr),
        int(comm_buf_ptrs_dev),
        int(step_pad_ptrs_dev),
        int(group_ranks_row_ptr),
        int(comm_buf_offset),
        int(self_global_rank),
        int(self_local_index),
        int(slice_bytes),
        int(partition_bytes),
        int(chunk_bytes),
        int(step_base),
        WORLD_SIZE=int(world_size),
        MAX_CHUNKS=int(max_chunks_per_channel),
        GROUP_SIZE=int(group_size),
        GROUP_LANES=int(_next_pow2(group_size)),
        NUM_CHANNELS=int(num_channels),
        BLOCK_ELTS=int(_DEFAULT_BLOCK_ELTS),
        VEC_BYTES=int(vec_bytes),
        TIMEOUT_NS=int(timeout_ns),
        num_warps=_DEFAULT_NUM_WARPS,
        num_stages=2,
    )


def launch_all2all_fused(
    *,
    input_ptr: int,
    output_ptr: int,
    comm_buf_ptrs_dev: int,
    step_pad_ptrs_dev: int,
    group_ranks_row_ptr: int,
    comm_buf_offset: int,
    self_global_rank: int,
    self_local_index: int,
    slice_bytes: int,
    group_size: int,
    world_size: int,
    max_chunks_per_channel: int,
    num_channels: int,
    chunk_bytes: int,
    step_base: int,
    alignment: int = 16,
    timeout_ns: int = 0,
) -> None:
    """Host-side launch of the fused all2all kernel."""
    assert slice_bytes > 0
    assert group_size > 0
    assert num_channels > 0
    assert chunk_bytes > 0

    partition_bytes = (slice_bytes + num_channels - 1) // num_channels
    partition_bytes = (partition_bytes + alignment - 1) & ~(alignment - 1)

    n_chunks = (partition_bytes + chunk_bytes - 1) // chunk_bytes
    if n_chunks > max_chunks_per_channel:
        # Hard check (not ``assert`` — must survive ``python -O``): exceeding
        # the static bound would index ``step_pad``'s chunk axis past its
        # allocation, an out-of-bounds write into symmetric memory.
        raise ValueError(
            f"per-channel chunk count {n_chunks} exceeds static bound "
            f"{max_chunks_per_channel}; raise fused_max_chunks_per_channel or "
            f"fused_chunk_size"
        )

    vec_bytes = _pick_vec_bytes(input_ptr, output_ptr, slice_bytes)

    grid = (2 * num_channels,)
    all2all_fused_kernel[grid](
        int(input_ptr),
        int(output_ptr),
        int(comm_buf_ptrs_dev),
        int(step_pad_ptrs_dev),
        int(group_ranks_row_ptr),
        int(comm_buf_offset),
        int(self_global_rank),
        int(self_local_index),
        int(slice_bytes),
        int(partition_bytes),
        int(chunk_bytes),
        int(step_base),
        WORLD_SIZE=int(world_size),
        MAX_CHUNKS=int(max_chunks_per_channel),
        GROUP_SIZE=int(group_size),
        GROUP_LANES=int(_next_pow2(group_size)),
        NUM_CHANNELS=int(num_channels),
        BLOCK_ELTS=int(_DEFAULT_BLOCK_ELTS),
        VEC_BYTES=int(vec_bytes),
        TIMEOUT_NS=int(timeout_ns),
        num_warps=_DEFAULT_NUM_WARPS,
        num_stages=2,
    )
