"""Shared helpers for kernels: vector-width selection, alignment, pointer packing."""

from __future__ import annotations

from math import gcd


def vec_width_bytes(*alignments: int) -> int:
    """Pick a per-launch vector width (1, 4, 8, or 16 bytes) given alignments.

    Inputs are byte alignments of every base address and the per-element
    payload size. The chosen width divides all of them.
    """
    align = 0
    for a in alignments:
        if a <= 0:
            return 1
        align = a if align == 0 else gcd(align, a)
    if align >= 16:
        return 16
    if align >= 8:
        return 8
    if align >= 4:
        return 4
    return 1


def base_alignment(addr: int) -> int:
    """Return the largest power-of-two that divides ``addr``."""
    if addr == 0:
        return 1 << 62
    return addr & (-addr)
