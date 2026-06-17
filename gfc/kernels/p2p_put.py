"""Remote-write data path for ``p2p_put``.

Functionally identical to ``pull_copy_kernel`` (it is a raw byte copy from
``src_ptr`` to ``dst_ptr``); the only difference is *where* the pointers
live. For ``p2p_get`` the source is a peer pointer and the destination is
local; for ``p2p_put`` the source is local (own comm_buf) and the destination
is a peer pointer. Triton emits a regular ``st.global.*`` for the store; on
NVLink P2P the peer pointer is a P2P-mapped virtual address so the store is
the remote write the spec calls out as "the only remote-write kernel".

This module re-exports the launcher for naming clarity.
"""

from __future__ import annotations

from gfc.kernels.copy_pull import launch_pull_copy as launch_p2p_put_copy

__all__ = ["launch_p2p_put_copy"]
