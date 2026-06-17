"""GroupDescriptor, stable hash, epoch manager, token computation.

The hash / epoch / token helpers are pure-Python with no GPU or distributed
dependencies. ``GroupDescriptor`` additionally carries an opaque reference to
the group's device-resident rank-list tensor (``ranks_dev``), allocated and
freed by the runtime; group.py itself never imports torch. See Section 2 of
the design spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from gfc.config import _require

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch dependency
    import torch

_U64_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def stable_hash64(*chunks: object) -> int:
    """Deterministic 64-bit hash seeded from FNV offset; chunks fold through splitmix64.

    Accepts bytes, str, ints, tuples, and lists. Used for group ids and
    session tokens. No external dependency; no Python ``hash()``.
    """
    h = 0xCBF29CE484222325
    for c in chunks:
        if isinstance(c, bytes):
            data: bytes = c
            for b in data:
                h = _splitmix64(h ^ b)
        elif isinstance(c, str):
            for b in c.encode():
                h = _splitmix64(h ^ b)
        elif isinstance(c, (tuple, list)):
            for v in c:
                h = _splitmix64(h ^ (int(v) & _U64_MASK))
        else:
            h = _splitmix64(h ^ (int(c) & _U64_MASK))
    return h


def compute_token(session_nonce: int, group_id: int, epoch: int) -> int:
    """Compute the 64-bit barrier token for ``(group_id, epoch)`` in this session.

    Forced non-zero: the signal cells are zero-initialised, so a 0 token would
    let the first acquire-spin on a freshly-used slot observe the init value and
    falsely pass the barrier. Remapping the single 0 pre-image to 1 keeps the
    full 64-bit range otherwise and guarantees the token never collides with the
    zero sentinel.
    """
    return stable_hash64(session_nonce, group_id, epoch) or 1


@dataclass(frozen=True)
class GroupDescriptor:
    group_id: int
    ranks: tuple[int, ...]
    local_index: int  # ranks.index(self_global_rank); -1 if not a member
    size: int
    # Device-resident ``uint32[size]`` ordered rank list, allocated by the
    # runtime at ``register_group`` and released at ``unregister_group``.
    # ``None`` for non-member descriptors (and in pure-Python unit tests).
    # Excluded from eq/hash: the handle's identity is its logical group, not
    # the backing tensor (which is also neither hashable nor ==-comparable).
    ranks_dev: Optional["torch.Tensor"] = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        _require(self.size == len(self.ranks), "GroupDescriptor.size must match ranks")
        _require(-1 <= self.local_index < self.size, "GroupDescriptor.local_index out of range")
        if self.local_index >= 0:
            _require(
                0 <= self.local_index < len(self.ranks),
                "GroupDescriptor.local_index out of range",
            )


class _EpochManager:
    """Strictly monotonic per-group epoch counter, host-side.

    Host increments are done at submission time, not at GPU completion.
    Stream FIFO order makes this correct.
    """

    def __init__(self) -> None:
        self._epochs: dict[int, int] = {}

    def next_for_barrier(self, gid: int) -> int:
        e = self._epochs.get(gid, 0)
        self._epochs[gid] = e + 1
        return e

    def next_pair_for_collective(self, gid: int) -> tuple[int, int]:
        e = self._epochs.get(gid, 0)
        self._epochs[gid] = e + 2
        return e, e + 1

    def peek(self, gid: int) -> int:
        """Return the next epoch that would be issued for ``gid`` (no advance)."""
        return self._epochs.get(gid, 0)

    def snapshot(self) -> dict[int, int]:
        return dict(self._epochs)
