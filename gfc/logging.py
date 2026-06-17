"""Logging helpers: INFO/DEBUG and the symmetric-memory footprint print."""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "gfc"


def get_logger() -> logging.Logger:
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("gfc[%(process)d] %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def _fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.1f} GiB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def log_footprint(
    *,
    max_collective_bytes: int,
    num_signal_slots: int,
    world_size: int,
    enable_debug_checks: bool,
    fused_num_channels: int = 0,
    fused_max_chunks_per_channel: int = 0,
) -> None:
    log = get_logger()
    comm_total = max_collective_bytes
    signal_total = num_signal_slots * world_size * 8
    step_pad_total = (
        fused_num_channels * fused_max_chunks_per_channel * world_size * 8
    )
    total = comm_total + signal_total + step_pad_total

    log.info("per-rank symmetric memory footprint")
    log.info(
        "  comm_buf  : %s",
        _fmt_bytes(max_collective_bytes),
    )
    log.info(
        "  signal_buf: %d slots x %d ranks x 8 B = %s",
        num_signal_slots,
        world_size,
        _fmt_bytes(signal_total),
    )
    log.info(
        "  step_pad  : %d ch x %d chunks x %d ranks x 8 B = %s   (used only by fused path)",
        fused_num_channels,
        fused_max_chunks_per_channel,
        world_size,
        _fmt_bytes(step_pad_total),
    )
    log.info(
        "  total per rank approx %s   (debug_checks=%s)",
        _fmt_bytes(total),
        enable_debug_checks,
    )
