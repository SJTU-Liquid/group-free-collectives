"""Small NVIDIA PTX helpers used by the Triton kernels."""

from __future__ import annotations

import triton.language as tl
from triton.language import core


@core.extern
def st_release_sys_u64(ptr, val, _semantic=None):
    """PTX ``st.global.release.sys.b64``.

    The signal grid has a single writer for each cell, so a release store is
    enough for publish; using an atomic RMW only adds traffic.
    """
    return tl.inline_asm_elementwise(
        asm="""
        st.global.release.sys.b64 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,l"),
        args=[
            ptr,
            tl.cast(val, tl.uint64, _semantic=_semantic),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def read_globaltimer(_semantic=None):
    """PTX ``mov.u64 %globaltimer`` — the device-side nanosecond wall clock.

    Used by the spin-wait watchdogs to bound how long a barrier / fused
    receiver may wait before aborting (see ``trap_if``).
    """
    return tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def trap_if(cond, _semantic=None):
    """Execute PTX ``trap`` (abort the kernel) on lanes where ``cond != 0``.

    The watchdog uses this to turn an otherwise-unbounded acquire spin into a
    hard, diagnosable failure: when the configured timeout elapses the kernel
    traps, so the host sees a CUDA error on the next sync instead of a GPU that
    hangs forever with no signal.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.ne.b32 %p0, $1, 0;
            @%p0 trap;
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,r"),
        args=[cond.to(tl.int32, _semantic=_semantic)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def ld_acquire_sys_u64(ptr, _semantic=None):
    """PTX ``ld.global.acquire.sys.b64``."""
    return tl.inline_asm_elementwise(
        asm="ld.global.acquire.sys.b64 $0, [$1];",
        constraints=("=l,l"),
        args=[ptr],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def st_release_sys_u64_pred(ptr, val, mask, _semantic=None):
    """Predicated ``st.global.release.sys.b64``: write ``val`` to ``ptr`` only
    on lanes where ``mask != 0``. Lanes with mask=0 emit no traffic.

    Used by the fused kernel to publish the step counter to every peer
    cell except this rank's own cell.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.ne.b32 %p0, $3, 0;
            @%p0 st.global.release.sys.b64 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,l,r"),
        args=[
            ptr,
            tl.cast(val, tl.uint64, _semantic=_semantic),
            mask.to(tl.int32, _semantic=_semantic),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def fence_acq_rel_sys(_semantic=None):
    """PTX ``fence.acq_rel.sys`` — a full acquire-release fence at system
    scope. Returns an int dummy (Triton requires a return value)."""
    return tl.inline_asm_elementwise(
        asm="""
        fence.acq_rel.sys;
        mov.u32 $0, 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def ld_global_v4_b32_cond(ptr, mask, _semantic=None):
    """Predicated 16-byte global load as four b32 registers."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.ne.b32 %p0, $5, 0;
            @%p0 ld.global.v4.b32 {$0, $1, $2, $3}, [$4];
        }
        """,
        constraints=("=r,=r,=r,=r,l,r"),
        args=[
            ptr,
            mask.to(tl.int32, _semantic=_semantic),
        ],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def st_global_v4_b32_cond(ptr, val0, val1, val2, val3, mask, _semantic=None):
    """Predicated 16-byte global store from four b32 registers."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.ne.b32 %p0, $6, 0;
            @%p0 st.global.v4.b32 [$1], {$2, $3, $4, $5};
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,r,r,r,r,r"),
        args=[
            ptr,
            tl.cast(val0, tl.uint32, _semantic=_semantic),
            tl.cast(val1, tl.uint32, _semantic=_semantic),
            tl.cast(val2, tl.uint32, _semantic=_semantic),
            tl.cast(val3, tl.uint32, _semantic=_semantic),
            mask.to(tl.int32, _semantic=_semantic),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )
