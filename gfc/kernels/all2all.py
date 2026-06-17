"""``all2all_pull_kernel`` — every rank pulls its slice from every peer.

Sender layout: rank ``s``'s ``comm_buf`` is partitioned into
``group.size`` equal ``slice_bytes`` chunks; slice ``t`` is destined for
``group.ranks[t]``.

Receiver: for every peer ``i`` in the group, pull slice
``self_local_index`` from that peer's comm_buf into
``output[i * slice_bytes : (i + 1) * slice_bytes]``.

Section 4.4 of the design spec.
"""

from __future__ import annotations

import triton
import triton.language as tl

from gfc.kernels._ptx import ld_global_v4_b32_cond, st_global_v4_b32_cond


@triton.jit
def all2all_pull_kernel(
    comm_buf_ptrs_dev_u64,
    dst_u64,
    group_ranks_row_u64,
    comm_buf_offset,             # int64: 0 for one-shot, chunk offset when chunked
    self_local_index,             # int32
    copy_bytes,                   # int64: bytes-per-peer to copy this launch
    src_slice_stride,             # int64: per-peer stride inside the sender's comm_buf (full slice_bytes)
    dst_stride_bytes,             # int64: per-peer stride in the destination buffer
    DTYPE: tl.constexpr,
    ELT_BYTES: tl.constexpr,
    BLOCK_ELTS: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group_ranks_ptr = group_ranks_row_u64.to(tl.pointer_type(tl.uint32))
    cb_ptrs = comm_buf_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))

    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64)

    n_full = copy_bytes // ELT_BYTES
    n_tiles = tl.cdiv(n_full, BLOCK_ELTS)
    total_tiles = GROUP_SIZE * n_tiles

    task = pid
    while task < total_tiles:
        tile_pid = task // GROUP_SIZE
        peer_delta = task - tile_pid * GROUP_SIZE
        # Ring-swizzle peer order by local rank. Without this, every rank
        # starts at peer 0 and creates avoidable hotspotting.
        peer_idx = (peer_delta + self_local_index + 1) % GROUP_SIZE
        peer = tl.load(group_ranks_ptr + peer_idx)

        peer_base = tl.load(cb_ptrs + peer)
        src_off = tl.cast(self_local_index, tl.int64) * tl.cast(src_slice_stride, tl.int64)
        src_base = peer_base + tl.cast(comm_buf_offset, tl.int64) + src_off
        dst_off = tl.cast(peer_idx, tl.int64) * tl.cast(dst_stride_bytes, tl.int64)
        dst_base = dst_u64 + dst_off

        base_elt = tile_pid * BLOCK_ELTS
        offs = base_elt + tl.arange(0, BLOCK_ELTS).to(tl.int64)
        mask = offs < n_full

        if ELT_BYTES == 16:
            byte_offs = offs * 16
            src_u8 = src_base.to(tl.pointer_type(tl.uint8)) + byte_offs
            dst_u8 = dst_base.to(tl.pointer_type(tl.uint8)) + byte_offs
            v0, v1, v2, v3 = ld_global_v4_b32_cond(src_u8, mask)
            st_global_v4_b32_cond(dst_u8, v0, v1, v2, v3, mask)
        else:
            src_p = src_base.to(tl.pointer_type(DTYPE))
            dst_p = dst_base.to(tl.pointer_type(DTYPE))

            x = tl.load(src_p + offs, mask=mask)
            tl.store(dst_p + offs, x, mask=mask)

        task += stride

    if HAS_TAIL:
        if pid < GROUP_SIZE:
            peer_idx = (pid + self_local_index + 1) % GROUP_SIZE
            peer = tl.load(group_ranks_ptr + peer_idx)
            peer_base = tl.load(cb_ptrs + peer)
            src_off = tl.cast(self_local_index, tl.int64) * tl.cast(src_slice_stride, tl.int64)
            src_base = peer_base + tl.cast(comm_buf_offset, tl.int64) + src_off
            dst_off = tl.cast(peer_idx, tl.int64) * tl.cast(dst_stride_bytes, tl.int64)
            dst_base = dst_u64 + dst_off

            tail_base = n_full * ELT_BYTES
            tail_offs = tail_base + tl.arange(0, ELT_BYTES).to(tl.int64)
            tail_mask = tail_offs < copy_bytes
            src_u8 = src_base.to(tl.pointer_type(tl.uint8))
            dst_u8 = dst_base.to(tl.pointer_type(tl.uint8))
            y = tl.load(src_u8 + tail_offs, mask=tail_mask)
            tl.store(dst_u8 + tail_offs, y, mask=tail_mask)


# -----------------------------------------------------------------------------
# Host launcher
# -----------------------------------------------------------------------------

_DEFAULT_BLOCK_BYTES = 64 * 1024


def _dtype_for_vec(vec_bytes: int):
    if vec_bytes >= 16:
        return tl.uint8, 16
    if vec_bytes >= 8:
        return tl.uint64, 8
    if vec_bytes == 4:
        return tl.uint32, 4
    return tl.uint8, 1


def launch_all2all_pull(
    *,
    comm_buf_ptrs_dev: int,
    dst_ptr: int,
    group_ranks_row_ptr: int,
    comm_buf_offset: int,
    self_local_index: int,
    slice_bytes: int,
    group_size: int,
    vec_bytes: int,
    num_sms: int = 24,
    block_bytes: int = _DEFAULT_BLOCK_BYTES,
    copy_bytes: int | None = None,
    src_slice_stride: int | None = None,
    dst_stride_bytes: int | None = None,
) -> None:
    """Pull-model all2all kernel launch.

    On the one-shot path ``slice_bytes`` is both the per-peer copy size and
    the per-peer stride. On the pipelined path the caller passes a smaller
    ``copy_bytes`` (chunk size) and keeps ``src_slice_stride`` /
    ``dst_stride_bytes`` equal to the full per-peer slice. The caller also
    folds ``chunk_offset_within_slice`` into both ``comm_buf_offset`` and
    ``dst_ptr`` so the kernel sees the chunk's base.
    """
    if copy_bytes is None:
        copy_bytes = slice_bytes
    if src_slice_stride is None:
        src_slice_stride = slice_bytes
    if dst_stride_bytes is None:
        dst_stride_bytes = slice_bytes
    assert copy_bytes > 0
    assert src_slice_stride >= copy_bytes
    assert dst_stride_bytes >= copy_bytes
    assert group_size > 0
    assert num_sms > 0
    dtype, elt_bytes = _dtype_for_vec(vec_bytes)
    n_full = copy_bytes // elt_bytes
    has_tail = (copy_bytes % elt_bytes) != 0
    block_elts = max(block_bytes // elt_bytes, 1)
    if n_full > 0:
        n_block = (n_full + block_elts - 1) // block_elts
    else:
        n_block = 0
    total_tasks = int(group_size) * int(n_block)
    if has_tail:
        total_tasks = max(total_tasks, int(group_size))
    else:
        total_tasks = max(total_tasks, 1)
    grid = (int(min(num_sms, total_tasks)),)

    all2all_pull_kernel[grid](
        int(comm_buf_ptrs_dev),
        int(dst_ptr),
        int(group_ranks_row_ptr),
        int(comm_buf_offset),
        int(self_local_index),
        int(copy_bytes),
        int(src_slice_stride),
        int(dst_stride_bytes),
        DTYPE=dtype,
        ELT_BYTES=elt_bytes,
        BLOCK_ELTS=block_elts,
        HAS_TAIL=has_tail,
        GROUP_SIZE=int(group_size),
        num_warps=4,
        num_stages=2,
    )
