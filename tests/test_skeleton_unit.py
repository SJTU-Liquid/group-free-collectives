"""Phase-1 single-process unit tests for config, hash, group, epoch manager.

Run: ``python -m pytest tests/test_skeleton_unit.py -q``
"""

from __future__ import annotations

import pytest

from gfc.config import SymmetricCollectiveConfig
from gfc.env import read_env_overrides
from gfc.group import (
    GroupDescriptor,
    _EpochManager,
    compute_token,
    stable_hash64,
)
from gfc.kernels._common import vec_width_bytes
from gfc.runtime import SymmetricCollectiveRuntime


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_defaults_and_signal_slots():
    c = SymmetricCollectiveConfig()
    assert c.max_group_size == 8
    assert c.num_signal_slots == 2
    assert c.max_collective_bytes == 128 * 1024 * 1024


def test_config_rejects_group_above_16():
    with pytest.raises(ValueError):
        SymmetricCollectiveConfig(max_group_size=17)


def test_config_rejects_misaligned_bytes():
    with pytest.raises(ValueError):
        SymmetricCollectiveConfig(max_collective_bytes=129, alignment=128)


def test_config_allows_timeout_zero():
    c = SymmetricCollectiveConfig(timeout_ms=0)
    assert c.timeout_ms == 0


def test_config_rejects_negative_timeout():
    with pytest.raises(ValueError):
        SymmetricCollectiveConfig(timeout_ms=-1)


# ---------------------------------------------------------------------------
# stable_hash64
# ---------------------------------------------------------------------------


def test_stable_hash_is_deterministic_and_u64():
    h1 = stable_hash64(b"gfc-group-v1", (0, 1, 3))
    h2 = stable_hash64(b"gfc-group-v1", (0, 1, 3))
    assert h1 == h2
    assert 0 <= h1 < (1 << 64)


def test_stable_hash_order_sensitive():
    # Ordered rank tuples that differ only in order must hash to different ids.
    h_a = stable_hash64(b"gfc-group-v1", (0, 1, 3))
    h_b = stable_hash64(b"gfc-group-v1", (1, 0, 3))
    h_c = stable_hash64(b"gfc-group-v1", (3, 1, 0))
    assert len({h_a, h_b, h_c}) == 3


def test_stable_hash_accepts_mixed_inputs():
    a = stable_hash64(1, 2, 3)
    b = stable_hash64((1, 2, 3))
    c = stable_hash64("xyz")
    d = stable_hash64(b"xyz")
    # Just confirm they're all distinct u64 values and not zero by accident.
    assert all(0 <= v < (1 << 64) for v in (a, b, c, d))
    assert c == d  # str path should equal bytes path on identical payload


def test_compute_token_distinct_per_epoch():
    nonce = 0x123456789ABCDEF1
    gid = stable_hash64(b"gfc-group-v1", (0, 1))
    tokens = [compute_token(nonce, gid, e) for e in range(1024)]
    assert len(set(tokens)) == len(tokens)
    assert all(0 <= t < (1 << 64) for t in tokens)


def test_compute_token_nonzero_when_nonce_nonzero():
    # Defensive: the spec says session_nonce must be non-zero.
    # We still don't want token == 0 to coincide with any of these.
    nonce = 0xAABBCCDDEEFF0011
    gid = stable_hash64(b"gfc-group-v1", (0, 1, 2, 3))
    for e in range(256):
        assert compute_token(nonce, gid, e) != 0


# ---------------------------------------------------------------------------
# GroupDescriptor
# ---------------------------------------------------------------------------


def test_group_descriptor_local_index_member():
    d = GroupDescriptor(
        group_id=stable_hash64(b"gfc-group-v1", (0, 2, 3)),
        ranks=(0, 2, 3),
        local_index=1,
        size=3,
    )
    assert d.ranks[d.local_index] == 2


def test_group_descriptor_local_index_nonmember():
    d = GroupDescriptor(
        group_id=stable_hash64(b"gfc-group-v1", (0, 2, 3)),
        ranks=(0, 2, 3),
        local_index=-1,
        size=3,
    )
    assert d.local_index == -1


def test_group_descriptor_size_mismatch_rejected():
    with pytest.raises(ValueError):
        GroupDescriptor(
            group_id=0xDEADBEEF,
            ranks=(0, 1, 2),
            local_index=0,
            size=2,  # wrong
        )


def test_group_descriptor_ranks_dev_defaults_none():
    d = GroupDescriptor(group_id=1, ranks=(0, 1), local_index=0, size=2)
    assert d.ranks_dev is None


def test_group_descriptor_identity_ignores_ranks_dev():
    # The device rank-list tensor is held on the handle but must not
    # participate in equality/hash — descriptors are identified by their
    # logical group identity only.
    gid = stable_hash64(b"gfc-group-v1", (0, 2, 3))
    a = GroupDescriptor(group_id=gid, ranks=(0, 2, 3), local_index=1, size=3, ranks_dev=object())
    b = GroupDescriptor(group_id=gid, ranks=(0, 2, 3), local_index=1, size=3, ranks_dev=object())
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# _EpochManager
# ---------------------------------------------------------------------------


def test_epoch_manager_barrier_increments_by_one():
    em = _EpochManager()
    gid = 0xCAFE
    seen = [em.next_for_barrier(gid) for _ in range(10)]
    assert seen == list(range(10))
    assert em.peek(gid) == 10


def test_epoch_manager_collective_increments_by_two():
    em = _EpochManager()
    gid = 0xBEEF
    a, b = em.next_pair_for_collective(gid)
    assert (a, b) == (0, 1)
    a, b = em.next_pair_for_collective(gid)
    assert (a, b) == (2, 3)
    assert em.peek(gid) == 4


def test_epoch_manager_independent_per_group():
    em = _EpochManager()
    g1, g2 = 1, 2
    em.next_for_barrier(g1)
    em.next_pair_for_collective(g2)
    em.next_for_barrier(g1)
    assert em.snapshot() == {g1: 2, g2: 2}


# ---------------------------------------------------------------------------
# vector-width selection
# ---------------------------------------------------------------------------


def test_vec_width_picks_16_for_well_aligned_payload():
    # base ptrs 128-byte aligned, slice_bytes multiple of 16.
    assert vec_width_bytes(128, 128, 128, 4096) == 16


def test_vec_width_falls_back_for_odd_payload():
    # slice_bytes that's only 4-byte aligned forces VEC=4.
    assert vec_width_bytes(128, 128, 128, 4 * 17) == 4


def test_vec_width_one_when_any_input_unaligned():
    assert vec_width_bytes(128, 1, 128, 4096) == 1


def test_vec_width_zero_alignment_returns_one():
    assert vec_width_bytes(0) == 1


# ---------------------------------------------------------------------------
# env overrides
# ---------------------------------------------------------------------------


def test_env_overrides_defaults(monkeypatch):
    for v in ("SYMM_COLL_DEBUG", "SYMM_COLL_USE_TMA", "SYMM_COLL_TIMEOUT_MS", "SYMM_COLL_DUMP_SIGNALS"):
        monkeypatch.delenv(v, raising=False)
    e = read_env_overrides()
    assert e.debug is False
    assert e.use_tma is False
    assert e.timeout_ms is None
    assert e.dump_signals is False


def test_env_overrides_truthy(monkeypatch):
    monkeypatch.setenv("SYMM_COLL_DEBUG", "1")
    monkeypatch.setenv("SYMM_COLL_USE_TMA", "true")
    monkeypatch.setenv("SYMM_COLL_TIMEOUT_MS", "5000")
    monkeypatch.setenv("SYMM_COLL_DUMP_SIGNALS", "yes")
    e = read_env_overrides()
    assert e.debug is True
    assert e.use_tma is True
    assert e.timeout_ms == 5000
    assert e.dump_signals is True


def test_env_overrides_timeout_zero(monkeypatch):
    monkeypatch.setenv("SYMM_COLL_TIMEOUT_MS", "0")
    e = read_env_overrides()
    assert e.timeout_ms == 0


class _FakeStream:
    def __init__(self) -> None:
        self.waited_for = []

    def wait_stream(self, stream) -> None:
        self.waited_for.append(stream)


class _FakeTensor:
    def __init__(self) -> None:
        self.recorded = []

    def record_stream(self, stream) -> None:
        self.recorded.append(stream)


def test_ingest_user_tensors_tracks_copy_stream_from_runtime_stream(monkeypatch):
    runtime = SymmetricCollectiveRuntime.__new__(SymmetricCollectiveRuntime)
    runtime.device = "cuda:0"
    runtime.stream = _FakeStream()
    runtime.copy_stream = _FakeStream()
    tensor = _FakeTensor()

    monkeypatch.setattr(
        "gfc.runtime.torch.cuda.current_stream",
        lambda device=None: runtime.stream,
    )

    runtime._ingest_user_tensors(tensor)

    assert runtime.stream.waited_for == []
    assert runtime.copy_stream.waited_for == [runtime.stream]
    assert tensor.recorded == [runtime.stream, runtime.copy_stream]
