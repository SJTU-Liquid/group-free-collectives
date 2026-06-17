"""Configuration object for the symmetric-collective runtime.

Defaults match Section 1.3 of the design spec.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require(cond: bool, msg: str) -> None:
    """Validate a user-supplied invariant.

    Uses an explicit ``raise`` rather than ``assert`` so configuration and
    argument checks are NOT silently stripped under ``python -O`` /
    ``PYTHONOPTIMIZE`` — under which an invalid config/argument would otherwise
    reach a kernel launch and fault out of bounds instead of failing cleanly.
    """
    if not cond:
        raise ValueError(msg)


@dataclass
class SymmetricCollectiveConfig:
    max_group_size: int = 8
    max_collective_bytes: int = 128 * 1024 * 1024
    alignment: int = 128
    use_tma: bool = False
    use_copy_engine: bool = False
    copy_sms: int = 24
    enable_debug_checks: bool = False
    timeout_ms: int = 30_000  # 0 disables the in-kernel watchdog.
    # Pipelined-staging: split a collective's local stage + remote pull into
    # ``pipeline_chunks`` chunks and overlap the local copy of chunk k+1 with
    # the remote pull of chunk k. ``pipeline_chunks <= 1`` keeps the original
    # one-shot path. ``max_pipeline_chunks`` is a static launch-time ceiling.
    # Barrier signalling now uses per-edge double-buffer slots (see
    # ``barrier_kernel``); these chunk knobs do not affect barrier layout.
    pipeline_chunks: int = 1
    max_pipeline_chunks: int = 4
    # Minimum chunk bytes-per-peer before chunking kicks in; smaller payloads
    # are dominated by barrier launch overhead.
    pipeline_min_chunk_bytes: int = 256 * 1024
    # Fused all-in-one kernel path. When enabled, all_gather / all2all run as
    # a single persistent kernel with CTA-pair sender/receiver roles and a
    # per-channel step counter, eliminating chunk-level host barrier launches.
    # See ``kernels/fused.py`` for the kernel; ``step_pad`` is the signal grid.
    enable_fused_path: bool = False
    fused_chunk_size: int = 256 * 1024
    fused_num_channels: int = 24  # grid = 2 * fused_num_channels (sender + receiver CTA per channel)
    # Path to an autotune JSON config produced by
    # ``benchmarks/autotune_collectives.py``. When set, every all_gather /
    # all2all call looks up the best path + knobs for the call's
    # ``(collective, group_size, slice_bytes)`` tuple instead of using
    # the static config flags.
    autotune_config_path: str | None = None

    @property
    def fused_max_chunks_per_channel(self) -> int:
        """Static upper bound on chunks-per-channel for ``step_pad`` sizing.

        Worst case occurs when a single channel handles the entire
        ``max_collective_bytes``. Bump by ``+1`` for tail-rounding slack.
        """
        return (self.max_collective_bytes + self.fused_chunk_size - 1) // self.fused_chunk_size + 1

    def __post_init__(self) -> None:
        _require(
            0 < self.max_group_size <= 16,
            "max_group_size must be in (0, 16] (barrier kernel hard ceiling)",
        )
        _require(self.max_collective_bytes > 0, "max_collective_bytes must be > 0")
        _require(
            self.max_collective_bytes % self.alignment == 0,
            "max_collective_bytes must be a multiple of alignment",
        )
        _require(
            self.alignment >= 16 and (self.alignment & (self.alignment - 1)) == 0,
            "alignment must be a power of two >= 16",
        )
        _require(self.copy_sms > 0, "copy_sms must be > 0")
        _require(self.timeout_ms >= 0, "timeout_ms must be >= 0")
        _require(self.pipeline_chunks >= 1, "pipeline_chunks must be >= 1")
        _require(self.max_pipeline_chunks >= 1, "max_pipeline_chunks must be >= 1")
        _require(
            self.pipeline_chunks <= self.max_pipeline_chunks,
            "pipeline_chunks must be <= max_pipeline_chunks",
        )
        _require(self.pipeline_min_chunk_bytes > 0, "pipeline_min_chunk_bytes must be > 0")
        _require(self.fused_chunk_size > 0, "fused_chunk_size must be > 0")
        _require(
            self.fused_chunk_size % self.alignment == 0,
            "fused_chunk_size must be a multiple of alignment",
        )
        _require(self.fused_num_channels > 0, "fused_num_channels must be > 0")

    @property
    def num_signal_slots(self) -> int:
        # Per-edge double-buffered barrier protocol. The signal grid is
        # ``uint64[2 slots, world_size src]``: each pair has 2 cells that
        # rotate per pair-local sequence parity, so consecutive barriers
        # between the same pair land on different slots. Tokens carry
        # group/epoch identity. There is no separate finish/ack lane —
        # slot reuse is delayed by one barrier and protected by pairwise
        # stream ordering. See ``gfc/kernels/barrier.py``.
        return 2
