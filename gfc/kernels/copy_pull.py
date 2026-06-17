"""``pull_copy_kernel`` — raw-byte pull copy from a peer to a local destination.

Used by ``p2p_get`` and as a building block by the higher-level kernels.
Section 4 of the design spec.
"""

from __future__ import annotations

import triton
import triton.language as tl

from gfc.kernels._ptx import ld_global_v4_b32_cond, st_global_v4_b32_cond


# -----------------------------------------------------------------------------
# Triton kernel
# -----------------------------------------------------------------------------


@triton.jit
def pull_copy_kernel(
    src_u64,                    # int64-packed peer base address
    dst_u64,                    # int64-packed local destination address
    nbytes,                     # tl.int64
    DTYPE: tl.constexpr,        # tl.uint8 / tl.uint32 / tl.uint64
    ELT_BYTES: tl.constexpr,    # 1, 4, or 8
    BLOCK_ELTS: tl.constexpr,   # elements per program
    HAS_TAIL: tl.constexpr,     # True iff nbytes is not a multiple of ELT_BYTES
):
    """Pull ``nbytes`` bytes from the peer-resident base at ``src_u64`` into the
    local destination at ``dst_u64``. Each program owns ``BLOCK_ELTS`` elements
    of size ``ELT_BYTES``. Optionally, the last program handles a per-byte tail.
    """
    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64)

    # ---- Phase 1: vector elements ------------------------------------------
    n_full = nbytes // ELT_BYTES                       # int64
    n_tiles = tl.cdiv(n_full, BLOCK_ELTS)
    tile_pid = pid
    while tile_pid < n_tiles:
        base_elt = tile_pid * BLOCK_ELTS
        offs = base_elt + tl.arange(0, BLOCK_ELTS).to(tl.int64)
        mask = offs < n_full

        if ELT_BYTES == 16:
            byte_offs = offs * 16
            src_u8 = src_u64.to(tl.pointer_type(tl.uint8)) + byte_offs
            dst_u8 = dst_u64.to(tl.pointer_type(tl.uint8)) + byte_offs
            v0, v1, v2, v3 = ld_global_v4_b32_cond(src_u8, mask)
            st_global_v4_b32_cond(dst_u8, v0, v1, v2, v3, mask)
        else:
            src_p = src_u64.to(tl.pointer_type(DTYPE))
            dst_p = dst_u64.to(tl.pointer_type(DTYPE))

            x = tl.load(src_p + offs, mask=mask)
            tl.store(dst_p + offs, x, mask=mask)

        tile_pid += stride

    # ---- Phase 2: per-byte tail (last program only) ------------------------
    if HAS_TAIL:
        if pid == 0:
            tail_base = n_full * ELT_BYTES
            tail_offs = tail_base + tl.arange(0, ELT_BYTES).to(tl.int64)
            tail_mask = tail_offs < nbytes
            src_u8 = src_u64.to(tl.pointer_type(tl.uint8))
            dst_u8 = dst_u64.to(tl.pointer_type(tl.uint8))
            y = tl.load(src_u8 + tail_offs, mask=tail_mask)
            tl.store(dst_u8 + tail_offs, y, mask=tail_mask)


# -----------------------------------------------------------------------------
# Host launcher
# -----------------------------------------------------------------------------


# Default tile size: 64 KiB per persistent program iteration. This keeps the
# loop grain coarse; parallelism comes from a fixed number of CTAs, not from
# launching a CTA per tiny tile.
_DEFAULT_BLOCK_BYTES = 64 * 1024


def _dtype_for_vec(vec_bytes: int):
    if vec_bytes >= 16:
        return tl.uint8, 16
    if vec_bytes >= 8:
        return tl.uint64, 8
    if vec_bytes == 4:
        return tl.uint32, 4
    return tl.uint8, 1


def launch_pull_copy(
    *,
    src_ptr: int,
    dst_ptr: int,
    nbytes: int,
    vec_bytes: int,
    num_sms: int = 24,
    block_bytes: int = _DEFAULT_BLOCK_BYTES,
) -> None:
    """Launch ``pull_copy_kernel`` to copy ``nbytes`` bytes from ``src_ptr`` to
    ``dst_ptr``. Uses the current CUDA stream — callers must enter the runtime
    stream context first (``with torch.cuda.stream(runtime.stream):``).
    """
    assert nbytes > 0, "nbytes must be positive"
    assert vec_bytes in (1, 4, 8, 16), f"unsupported vec_bytes={vec_bytes}"
    assert num_sms > 0, "num_sms must be positive"

    dtype, elt_bytes = _dtype_for_vec(vec_bytes)
    n_full = nbytes // elt_bytes
    has_tail = (nbytes % elt_bytes) != 0
    block_elts = max(block_bytes // elt_bytes, 1)
    if n_full > 0:
        grid_elems = (n_full + block_elts - 1) // block_elts
    else:
        # All bytes go to the tail; still launch one program to perform it.
        grid_elems = 1
    grid = (int(min(num_sms, grid_elems)),)

    pull_copy_kernel[grid](
        int(src_ptr),
        int(dst_ptr),
        int(nbytes),
        DTYPE=dtype,
        ELT_BYTES=elt_bytes,
        BLOCK_ELTS=block_elts,
        HAS_TAIL=has_tail,
        num_warps=4,
        num_stages=2,
    )
