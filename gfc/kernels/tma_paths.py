"""TMA-variant kernels: persistent loop + large tile.

Each launch spins up ``NUM_SMS`` programs (matching the device SM count);
every program iterates ``pid += NUM_SMS`` over the total tile count. This
amortises per-tile launch / scheduling jitter and keeps every SM warm
across the whole collective. Tile size is chosen to fill shared memory
backing the TMA descriptor (H100 has 228 KiB / SM; we use 64 KiB so
double-buffered pipelines still fit).

For all_gather and all2all we serialise the (peer, tile) pair into a
single 1-D persistent index; the descriptor for the active peer is built
inside the kernel via :func:`tl.make_tensor_descriptor`. Each iteration
rebuilds the descriptor — the build is cheap relative to the 64 KiB TMA
load it amortises over.

For ``p2p_put`` and ``p2p_get`` there is exactly one peer per call, so
the descriptor is built once at the top of the persistent loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gfc.tma_probe import TMARequirementError


_TMA_ALIGN = 16
_DEFAULT_TILE_BYTES = 64 * 1024


def _device_num_sms(device: torch.device | None = None) -> int:
    if device is None:
        device = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def check_collective_tma_gates(
    *, slice_bytes: int, comm_buf_offset: int, dst_ptr: int, tile_bytes: int = _DEFAULT_TILE_BYTES
) -> None:
    """Validate per-call TMA requirements for collective launches.

    Runtime dispatch calls this before consuming epochs / issuing pre-barriers;
    launchers call it again as a defence-in-depth backstop.
    """
    if slice_bytes <= 0:
        raise TMARequirementError("slice_bytes must be positive")
    if slice_bytes % _TMA_ALIGN != 0:
        raise TMARequirementError(
            f"slice_bytes={slice_bytes} not a multiple of {_TMA_ALIGN}"
        )
    if comm_buf_offset % _TMA_ALIGN != 0 or dst_ptr % _TMA_ALIGN != 0:
        raise TMARequirementError(
            f"TMA needs 16B-aligned bases (comm_buf_off={comm_buf_offset}, dst={dst_ptr:#x})"
        )
    if tile_bytes % _TMA_ALIGN != 0:
        raise TMARequirementError(
            f"tile_bytes={tile_bytes} must be a multiple of {_TMA_ALIGN}"
        )


# -----------------------------------------------------------------------------
# all_gather (persistent over (peer_idx, tile_col))
# -----------------------------------------------------------------------------


@triton.jit
def _all_gather_tma_persistent_kernel(
    comm_buf_ptrs_dev_u64,
    dst_u64,
    group_ranks_row_u64,
    comm_buf_offset,
    self_local_index,
    slice_bytes,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    inner_tiles = tl.cdiv(slice_bytes, BLOCK_N)
    total = GROUP_SIZE * inner_tiles

    group_ranks_ptr = group_ranks_row_u64.to(tl.pointer_type(tl.uint32))
    cb_ptrs = comm_buf_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))

    while pid < total:
        # Tile-major task order keeps the first wave spread over peers. The
        # peer dimension is ring-swizzled by local rank to avoid all ranks
        # touching the same remote GPU first.
        tile_col = pid // GROUP_SIZE
        peer_delta = pid - tile_col * GROUP_SIZE
        peer_idx = (peer_delta + self_local_index + 1) % GROUP_SIZE
        col_start = tile_col * BLOCK_N

        peer = tl.load(group_ranks_ptr + peer_idx)
        peer_base = tl.load(cb_ptrs + peer)
        src_base = peer_base + tl.cast(comm_buf_offset, tl.int64)
        dst_off = tl.cast(peer_idx, tl.int64) * tl.cast(slice_bytes, tl.int64)
        dst_base = dst_u64 + dst_off

        src_desc = tl.make_tensor_descriptor(
            src_base.to(tl.pointer_type(tl.uint8)),
            shape=[slice_bytes],
            strides=[1],
            block_shape=[BLOCK_N],
        )
        dst_desc = tl.make_tensor_descriptor(
            dst_base.to(tl.pointer_type(tl.uint8)),
            shape=[slice_bytes],
            strides=[1],
            block_shape=[BLOCK_N],
        )

        tile = src_desc.load([col_start])
        dst_desc.store([col_start], tile)

        pid += NUM_SMS


def launch_all_gather_tma(
    *,
    comm_buf_ptrs_dev: int,
    dst_ptr: int,
    group_ranks_row_ptr: int,
    comm_buf_offset: int,
    self_local_index: int,
    slice_bytes: int,
    group_size: int,
    num_sms: int,
    tile_bytes: int = _DEFAULT_TILE_BYTES,
) -> None:
    check_collective_tma_gates(
        slice_bytes=slice_bytes,
        comm_buf_offset=comm_buf_offset,
        dst_ptr=dst_ptr,
        tile_bytes=tile_bytes,
    )

    inner_tiles = (slice_bytes + tile_bytes - 1) // tile_bytes
    total = group_size * inner_tiles
    grid = (min(num_sms, total),)

    _all_gather_tma_persistent_kernel[grid](
        int(comm_buf_ptrs_dev),
        int(dst_ptr),
        int(group_ranks_row_ptr),
        int(comm_buf_offset),
        int(self_local_index),
        int(slice_bytes),
        GROUP_SIZE=int(group_size),
        BLOCK_N=int(tile_bytes),
        NUM_SMS=int(num_sms),
        num_warps=4,
        num_stages=2,
    )


# -----------------------------------------------------------------------------
# all2all (persistent over (peer_idx, tile_col)) — pulls slice [self_local_index]
# -----------------------------------------------------------------------------


@triton.jit
def _all2all_tma_persistent_kernel(
    comm_buf_ptrs_dev_u64,
    dst_u64,
    group_ranks_row_u64,
    comm_buf_offset,
    self_local_index,
    slice_bytes,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    inner_tiles = tl.cdiv(slice_bytes, BLOCK_N)
    total = GROUP_SIZE * inner_tiles

    group_ranks_ptr = group_ranks_row_u64.to(tl.pointer_type(tl.uint32))
    cb_ptrs = comm_buf_ptrs_dev_u64.to(tl.pointer_type(tl.uint64))

    while pid < total:
        # Same tile-major + ring peer order as the vector path.
        tile_col = pid // GROUP_SIZE
        peer_delta = pid - tile_col * GROUP_SIZE
        peer_idx = (peer_delta + self_local_index + 1) % GROUP_SIZE
        col_start = tile_col * BLOCK_N

        peer = tl.load(group_ranks_ptr + peer_idx)
        peer_base = tl.load(cb_ptrs + peer)
        src_off = tl.cast(self_local_index, tl.int64) * tl.cast(slice_bytes, tl.int64)
        src_base = peer_base + tl.cast(comm_buf_offset, tl.int64) + src_off
        dst_off = tl.cast(peer_idx, tl.int64) * tl.cast(slice_bytes, tl.int64)
        dst_base = dst_u64 + dst_off

        src_desc = tl.make_tensor_descriptor(
            src_base.to(tl.pointer_type(tl.uint8)),
            shape=[slice_bytes],
            strides=[1],
            block_shape=[BLOCK_N],
        )
        dst_desc = tl.make_tensor_descriptor(
            dst_base.to(tl.pointer_type(tl.uint8)),
            shape=[slice_bytes],
            strides=[1],
            block_shape=[BLOCK_N],
        )

        tile = src_desc.load([col_start])
        dst_desc.store([col_start], tile)

        pid += NUM_SMS


def launch_all2all_tma(
    *,
    comm_buf_ptrs_dev: int,
    dst_ptr: int,
    group_ranks_row_ptr: int,
    comm_buf_offset: int,
    self_local_index: int,
    slice_bytes: int,
    group_size: int,
    num_sms: int,
    tile_bytes: int = _DEFAULT_TILE_BYTES,
) -> None:
    check_collective_tma_gates(
        slice_bytes=slice_bytes,
        comm_buf_offset=comm_buf_offset,
        dst_ptr=dst_ptr,
        tile_bytes=tile_bytes,
    )

    inner_tiles = (slice_bytes + tile_bytes - 1) // tile_bytes
    total = group_size * inner_tiles
    grid = (min(num_sms, total),)

    _all2all_tma_persistent_kernel[grid](
        int(comm_buf_ptrs_dev),
        int(dst_ptr),
        int(group_ranks_row_ptr),
        int(comm_buf_offset),
        int(self_local_index),
        int(slice_bytes),
        GROUP_SIZE=int(group_size),
        BLOCK_N=int(tile_bytes),
        NUM_SMS=int(num_sms),
        num_warps=4,
        num_stages=2,
    )


# -----------------------------------------------------------------------------
# Single-peer copy (p2p_get pull side, p2p_put push side) — persistent over tile
# -----------------------------------------------------------------------------


@triton.jit
def _p2p_tma_persistent_kernel(
    src_u64,
    dst_u64,
    nbytes,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total = tl.cdiv(nbytes, BLOCK_N)

    src_desc = tl.make_tensor_descriptor(
        src_u64.to(tl.pointer_type(tl.uint8)),
        shape=[nbytes],
        strides=[1],
        block_shape=[BLOCK_N],
    )
    dst_desc = tl.make_tensor_descriptor(
        dst_u64.to(tl.pointer_type(tl.uint8)),
        shape=[nbytes],
        strides=[1],
        block_shape=[BLOCK_N],
    )

    while pid < total:
        off = pid * BLOCK_N
        tile = src_desc.load([off])
        dst_desc.store([off], tile)
        pid += NUM_SMS


def launch_p2p_copy_tma(
    *,
    src_ptr: int,
    dst_ptr: int,
    nbytes: int,
    num_sms: int,
    tile_bytes: int = _DEFAULT_TILE_BYTES,
) -> None:
    """Single-peer persistent TMA byte copy. Used by both p2p_put (src=local,
    dst=peer) and p2p_get (src=peer, dst=local) when TMA is enabled."""
    if nbytes <= 0:
        raise TMARequirementError("nbytes must be positive")
    if nbytes % _TMA_ALIGN != 0:
        raise TMARequirementError(
            f"nbytes={nbytes} not a multiple of {_TMA_ALIGN}"
        )
    if src_ptr % _TMA_ALIGN != 0 or dst_ptr % _TMA_ALIGN != 0:
        raise TMARequirementError(
            f"TMA needs 16B-aligned src/dst ({src_ptr:#x}, {dst_ptr:#x})"
        )
    if tile_bytes % _TMA_ALIGN != 0:
        raise TMARequirementError(
            f"tile_bytes={tile_bytes} must be a multiple of {_TMA_ALIGN}"
        )

    total = (nbytes + tile_bytes - 1) // tile_bytes
    grid = (min(num_sms, total),)

    _p2p_tma_persistent_kernel[grid](
        int(src_ptr),
        int(dst_ptr),
        int(nbytes),
        BLOCK_N=int(tile_bytes),
        NUM_SMS=int(num_sms),
        num_warps=4,
        num_stages=2,
    )
