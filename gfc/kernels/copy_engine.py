"""Host-side copy-engine data path via ``cudaMemcpyAsync``.

The pointers passed here are CUDA virtual addresses, including peer-mapped
addresses from PyTorch symmetric memory. ``cudaMemcpyDefault`` lets the driver
route local and peer copies through the appropriate device copy path.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Sequence

import torch


_CUDA_MEMCPY_DEFAULT = 4
_CUDART = None


def _load_cudart():
    global _CUDART
    if _CUDART is not None:
        return _CUDART

    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates.extend(("libcudart.so", "libcudart.so.13", "libcudart.so.12"))

    last_error: OSError | None = None
    for name in candidates:
        try:
            rt = ctypes.CDLL(name)
            rt.cudaMemcpyAsync.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            rt.cudaMemcpyAsync.restype = ctypes.c_int
            rt.cudaGetErrorString.argtypes = [ctypes.c_int]
            rt.cudaGetErrorString.restype = ctypes.c_char_p
            _CUDART = rt
            return rt
        except OSError as e:
            last_error = e

    raise RuntimeError(f"could not load libcudart: {last_error}")


def cuda_memcpy_async(
    *,
    dst_ptr: int,
    src_ptr: int,
    nbytes: int,
    stream: torch.cuda.Stream,
) -> None:
    """Enqueue ``cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault, stream)``."""
    if nbytes <= 0:
        return

    rt = _load_cudart()
    err = rt.cudaMemcpyAsync(
        ctypes.c_void_p(int(dst_ptr)),
        ctypes.c_void_p(int(src_ptr)),
        ctypes.c_size_t(int(nbytes)),
        ctypes.c_int(_CUDA_MEMCPY_DEFAULT),
        ctypes.c_void_p(int(stream.cuda_stream)),
    )
    if err != 0:
        msg = rt.cudaGetErrorString(err)
        text = msg.decode("utf-8", errors="replace") if msg else "unknown"
        raise RuntimeError(f"cudaMemcpyAsync failed with error {err}: {text}")


def _swizzled_indices(group_size: int, self_local_index: int) -> list[int]:
    return [
        (delta + self_local_index + 1) % group_size
        for delta in range(group_size)
    ]


def launch_all_gather_copy_engine(
    *,
    peer_comm_ptrs: Sequence[int],
    dst_ptr: int,
    group_ranks: Sequence[int],
    comm_buf_offset: int,
    self_local_index: int,
    slice_bytes: int,
    stream: torch.cuda.Stream,
) -> None:
    group_size = len(group_ranks)
    for peer_idx in _swizzled_indices(group_size, self_local_index):
        peer = int(group_ranks[peer_idx])
        src = int(peer_comm_ptrs[peer]) + int(comm_buf_offset)
        dst = int(dst_ptr) + peer_idx * int(slice_bytes)
        cuda_memcpy_async(dst_ptr=dst, src_ptr=src, nbytes=slice_bytes, stream=stream)


def launch_all2all_copy_engine(
    *,
    peer_comm_ptrs: Sequence[int],
    dst_ptr: int,
    group_ranks: Sequence[int],
    comm_buf_offset: int,
    self_local_index: int,
    slice_bytes: int,
    stream: torch.cuda.Stream,
) -> None:
    group_size = len(group_ranks)
    for peer_idx in _swizzled_indices(group_size, self_local_index):
        peer = int(group_ranks[peer_idx])
        src = (
            int(peer_comm_ptrs[peer])
            + int(comm_buf_offset)
            + int(self_local_index) * int(slice_bytes)
        )
        dst = int(dst_ptr) + peer_idx * int(slice_bytes)
        cuda_memcpy_async(dst_ptr=dst, src_ptr=src, nbytes=slice_bytes, stream=stream)


def launch_p2p_copy_engine(
    *,
    src_ptr: int,
    dst_ptr: int,
    nbytes: int,
    stream: torch.cuda.Stream,
) -> None:
    cuda_memcpy_async(dst_ptr=dst_ptr, src_ptr=src_ptr, nbytes=nbytes, stream=stream)
