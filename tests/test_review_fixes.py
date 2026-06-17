"""Regression coverage for the June-2026 code-review fix pass.

Each check maps to a finding from that review:

* #6 — ``register_group`` rejects a ``group_id`` reused with different ranks.
* #7 — ``all_gather`` / ``all2all`` reject inexact / mis-sized buffers.
* #3 — an autotune ``tma`` policy is rejected when TMA is not enabled.
* #1 — an autotune ``fused`` policy with more channels than the allocated
       ``step_pad`` is rejected (would otherwise OOB symmetric memory).
* #4 — the pipelined vec path stays correct when ``slice_bytes`` is not
       16-aligned (the per-peer stride is folded into the vec-width choice).
* #2 — the fused path stays correct under *overlapping* groups, where a
       following collective on a different group reuses ``comm_buf`` (the
       fused post-barrier prevents a peer-still-reading hazard). world>=3.

Plus a later pass:

* autotune/runtime mismatch — loading an autotune table whose fused policy
  needs more channels than the runtime allocated ``step_pad`` for fails fast at
  construction (with the knob to set) instead of mid-run on a large bucket.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_review_fixes.py
    torchrun --standalone --nproc_per_node=4 tests/test_review_fixes.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import dataclasses

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
)
from gfc.autotune import AutotuneTable  # noqa: E402

from tests._harness import rank_print, setup_nccl, teardown  # noqa: E402


def _catch_all_table(config: dict) -> AutotuneTable:
    return AutotuneTable(
        [{"collective": "*", "group_size": "*", "min_bytes": 0, "config": config}]
    )


def _expect_value_error(label: str, fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{label}: expected ValueError, none raised")


def _ag_pattern(rank: int, tag: int, nbytes: int, device) -> torch.Tensor:
    idx = torch.arange(nbytes, dtype=torch.int64, device=device)
    return (((tag + 1) * 131 + (rank + 1) * 17 + idx * 7) & 0xFF).to(torch.uint8)


# --------------------------------------------------------------------- #4
def _check_pipelined_unaligned(runtime: SymmetricCollectiveRuntime, group) -> None:
    """all_gather + all2all on the pipelined path with a non-16-aligned
    slice. Drives the path via an autotune ``pipelined`` policy so it runs
    regardless of the runtime's default path."""
    runtime.autotune = _catch_all_table({"path": "pipelined", "pipeline_chunks": 4})
    try:
        # 2056 = 8 * 257: 8-byte aligned, NOT 16-byte aligned. Big enough that
        # _effective_pipeline_chunks (min_chunk=1024) yields >1 chunk.
        slice_bytes = 2056
        gs = group.size

        # all_gather
        inp = _ag_pattern(runtime.rank, 1, slice_bytes, runtime.device)
        out = torch.empty(gs * slice_bytes, dtype=torch.uint8, device=runtime.device)
        runtime.all_gather(inp, out, group)
        runtime.stream.synchronize()
        for seg, peer in enumerate(group.ranks):
            exp = _ag_pattern(peer, 1, slice_bytes, runtime.device)
            got = out[seg * slice_bytes : (seg + 1) * slice_bytes]
            if not torch.equal(got, exp):
                raise AssertionError(
                    f"#4 all_gather mismatch seg={seg} peer={peer} "
                    f"slice_bytes={slice_bytes}"
                )

        # all2all
        a_in = torch.empty(gs * slice_bytes, dtype=torch.uint8, device=runtime.device)
        for t in range(gs):
            a_in[t * slice_bytes : (t + 1) * slice_bytes] = _ag_pattern(
                runtime.rank * 100 + t, 2, slice_bytes, runtime.device
            )
        a_out = torch.empty_like(a_in)
        runtime.all2all(a_in, a_out, group)
        runtime.stream.synchronize()
        for seg, peer in enumerate(group.ranks):
            exp = _ag_pattern(peer * 100 + group.local_index, 2, slice_bytes, runtime.device)
            got = a_out[seg * slice_bytes : (seg + 1) * slice_bytes]
            if not torch.equal(got, exp):
                raise AssertionError(
                    f"#4 all2all mismatch seg={seg} peer={peer} "
                    f"slice_bytes={slice_bytes}"
                )
    finally:
        runtime.autotune = None


# --------------------------------------------------------------------- #6/#7
def _check_arg_validation(runtime: SymmetricCollectiveRuntime, group) -> None:
    world = runtime.world_size
    dev = runtime.device

    # #6 — same explicit id, different ranks must be rejected.
    a = tuple(range(world))
    b = tuple(reversed(range(world)))
    gid = 0x7E57
    ga = runtime.register_group(a, group_id=gid)
    try:
        _expect_value_error(
            "#6 conflicting group_id",
            lambda: runtime.register_group(b, group_id=gid),
        )
    finally:
        runtime.unregister_group(ga)

    # #7 — all_gather output must be exactly group.size * input.nbytes (not
    # merely floor-divisible).
    inp = torch.zeros(1024, dtype=torch.uint8, device=dev)
    bad_out = torch.zeros(group.size * 1024 + 8, dtype=torch.uint8, device=dev)
    _expect_value_error(
        "#7 all_gather oversized output",
        lambda: runtime.all_gather(inp, bad_out, group),
    )

    # #7 — all2all explicit slice_bytes that doesn't tile the input.
    a2a_in = torch.zeros(group.size * 1024, dtype=torch.uint8, device=dev)
    a2a_out = torch.zeros_like(a2a_in)
    _expect_value_error(
        "#7 all2all bad slice_bytes",
        lambda: runtime.all2all(a2a_in, a2a_out, group, slice_bytes=1000),
    )


# --------------------------------------------------------------------- #1/#3
def _check_autotune_guards(runtime: SymmetricCollectiveRuntime, group) -> None:
    dev = runtime.device
    inp = torch.zeros(4096, dtype=torch.uint8, device=dev)
    out = torch.zeros(group.size * 4096, dtype=torch.uint8, device=dev)

    # #3 — tma policy rejected because the runtime was built with use_tma=False.
    assert not runtime.tma_enabled, "test assumes TMA disabled"
    runtime.autotune = _catch_all_table({"path": "tma"})
    try:
        _expect_value_error(
            "#3 autotune tma without probe",
            lambda: runtime.all_gather(inp, out, group),
        )
    finally:
        runtime.autotune = None

    # TMA per-call gates must also reject before reserving epochs. This is
    # forced without requiring TMA-capable hardware by toggling the runtime gate
    # only for an unsupported, misaligned payload that never reaches a kernel.
    bad_tma_inp = torch.zeros(4100, dtype=torch.uint8, device=dev)
    bad_tma_out = torch.zeros(group.size * 4100, dtype=torch.uint8, device=dev)
    before = runtime._epochs.peek(group.group_id)
    old_tma_enabled = runtime.tma_enabled
    runtime.tma_enabled = True
    runtime.autotune = _catch_all_table({"path": "tma"})
    try:
        _expect_value_error(
            "P2 autotune tma misaligned preflight",
            lambda: runtime.all_gather(bad_tma_inp, bad_tma_out, group),
        )
    finally:
        runtime.autotune = None
        runtime.tma_enabled = old_tma_enabled
    after = runtime._epochs.peek(group.group_id)
    assert before == after, "P2 tma preflight consumed epochs before rejecting"

    before = runtime._epochs.peek(group.group_id)
    old_tma_enabled = runtime.tma_enabled
    old_config = runtime.config
    runtime.tma_enabled = True
    runtime.config = dataclasses.replace(runtime.config, enable_fused_path=False)
    try:
        _expect_value_error(
            "P2 default tma misaligned preflight",
            lambda: runtime.all_gather(bad_tma_inp, bad_tma_out, group),
        )
    finally:
        runtime.config = old_config
        runtime.tma_enabled = old_tma_enabled
    after = runtime._epochs.peek(group.group_id)
    assert before == after, "P2 default tma preflight consumed epochs before rejecting"

    # #1 / P1 — bad fused knobs must be rejected, and rejected *before* any
    # epoch is consumed (a half-issued collective would perturb the per-edge
    # slot sequence). Each case asserts the per-group epoch is untouched.
    cfg = runtime.config
    bad_fused = [
        ("channel overflow", {"path": "fused", "fused_num_channels": cfg.fused_num_channels + 8}),
        ("zero channels", {"path": "fused", "fused_num_channels": 0}),
        ("non-aligned chunk_size", {"path": "fused", "fused_chunk_size": cfg.alignment + 1}),
        ("zero chunk_size", {"path": "fused", "fused_chunk_size": 0}),
    ]
    for label, knobs in bad_fused:
        before = runtime._epochs.peek(group.group_id)
        runtime.autotune = _catch_all_table(knobs)
        try:
            _expect_value_error(
                f"P1 fused {label}",
                lambda: runtime.all_gather(inp, out, group),
            )
        finally:
            runtime.autotune = None
        after = runtime._epochs.peek(group.group_id)
        assert before == after, (
            f"P1 fused {label}: guard consumed {after - before} epoch(s) "
            f"before rejecting"
        )

    # P1 — fused_chunk_size too small overflows the static chunk bound. A
    # min-alignment chunk over a single channel forces n_chunks past the bound.
    big = (cfg.fused_max_chunks_per_channel + 4) * cfg.alignment
    assert big <= cfg.max_collective_bytes
    big_in = torch.zeros(big, dtype=torch.uint8, device=dev)
    big_out = torch.zeros(group.size * big, dtype=torch.uint8, device=dev)
    before = runtime._epochs.peek(group.group_id)
    runtime.autotune = _catch_all_table(
        {"path": "fused", "fused_num_channels": 1, "fused_chunk_size": cfg.alignment}
    )
    try:
        _expect_value_error(
            "P1 fused chunk-count overflow",
            lambda: runtime.all_gather(big_in, big_out, group),
        )
    finally:
        runtime.autotune = None
    after = runtime._epochs.peek(group.group_id)
    assert before == after, "P1 chunk overflow: guard consumed epochs before rejecting"


def _check_autotune_fused_ok(runtime: SymmetricCollectiveRuntime, group) -> None:
    """A *valid* fused autotune policy (channels within the allocated bound)
    must run correctly through the dispatch path, including its post-barrier."""
    runtime.autotune = _catch_all_table(
        {
            "path": "fused",
            "fused_num_channels": runtime.config.fused_num_channels,
            "fused_chunk_size": 256 * 1024,
        }
    )
    try:
        slice_bytes = 4096
        gs = group.size
        inp = _ag_pattern(runtime.rank, 7, slice_bytes, runtime.device)
        out = torch.empty(gs * slice_bytes, dtype=torch.uint8, device=runtime.device)
        runtime.all_gather(inp, out, group)
        runtime.stream.synchronize()
        for seg, peer in enumerate(group.ranks):
            exp = _ag_pattern(peer, 7, slice_bytes, runtime.device)
            got = out[seg * slice_bytes : (seg + 1) * slice_bytes]
            if not torch.equal(got, exp):
                raise AssertionError(
                    f"autotune fused all_gather mismatch seg={seg} peer={peer}"
                )
    finally:
        runtime.autotune = None


def _check_fused_all2all_multichunk(runtime: SymmetricCollectiveRuntime, group) -> None:
    """Drive fused all2all through multiple chunks per channel."""
    runtime.autotune = _catch_all_table(
        {
            "path": "fused",
            "fused_num_channels": 1,
            "fused_chunk_size": 1024,
        }
    )
    try:
        slice_bytes = 4096
        gs = group.size
        a_in = torch.empty(gs * slice_bytes, dtype=torch.uint8, device=runtime.device)
        for t in range(gs):
            a_in[t * slice_bytes : (t + 1) * slice_bytes] = _ag_pattern(
                runtime.rank * 100 + t, 11, slice_bytes, runtime.device
            )
        a_out = torch.empty_like(a_in)
        runtime.all2all(a_in, a_out, group)
        runtime.stream.synchronize()
        for seg, peer in enumerate(group.ranks):
            exp = _ag_pattern(
                peer * 100 + group.local_index, 11, slice_bytes, runtime.device
            )
            got = a_out[seg * slice_bytes : (seg + 1) * slice_bytes]
            if not torch.equal(got, exp):
                raise AssertionError(
                    f"fused multichunk all2all mismatch seg={seg} peer={peer}"
                )
    finally:
        runtime.autotune = None


# ----------------------------------------------- autotune/runtime mismatch
def _check_autotune_load_validation(device) -> None:
    """An autotune table whose fused policy requests more channels than the
    runtime is provisioned for must be rejected at construction, not deep in a
    later large-payload dispatch. The check runs before any symmetric-memory
    rendezvous in ``__init__``, so every rank raises symmetrically (no hang).

    A channel count far above any realistic provisioning (256 → a 512-CTA grid)
    keeps the check independent of the default / env-overridden
    ``fused_num_channels``."""
    over = 256
    table = {
        "version": 1,
        "rules": [
            {
                "collective": "all2all",
                "group_size": "*",
                "min_bytes": 0,
                "config": {
                    "path": "fused",
                    "fused_num_channels": over,
                    "fused_chunk_size": 256 * 1024,
                },
            }
        ],
    }
    # Per-rank temp file (identical content) so there is no cross-rank read/write
    # race on a shared path.
    fd, path = tempfile.mkstemp(suffix=f"_gfc_autotune_r{dist.get_rank()}.json")
    with os.fdopen(fd, "w") as fh:
        json.dump(table, fh)
    try:
        cfg = SymmetricCollectiveConfig(
            max_group_size=max(dist.get_world_size(), 2),
            autotune_config_path=path,
        )
        _expect_value_error(
            "autotune over-channel load",
            lambda: SymmetricCollectiveRuntime(cfg, device=device),
        )
    finally:
        os.unlink(path)


# --------------------------------------------------------------------- #2
def _check_fused_overlapping_groups(runtime: SymmetricCollectiveRuntime) -> None:
    """world>=3: rank 0 runs (0,1) then (0,2) back-to-back on the fused path
    while rank 1 is still pulling from rank 0's comm_buf. The fused post-barrier
    must prevent rank 0 from overwriting comm_buf early. Verify correctness over
    many iterations."""
    rank = runtime.rank
    g01 = runtime.register_group((0, 1)) if rank in (0, 1) else None
    g02 = runtime.register_group((0, 2)) if rank in (0, 2) else None

    nbytes = 4096  # 16-aligned → fused vec=16
    n_iter = 200
    for it in range(n_iter):
        # Rank 0 issues (0,1) THEN (0,2) — the order from the finding.
        if rank in (0, 1):
            inp = _ag_pattern(rank, 1000 + it, nbytes, runtime.device)
            out01 = torch.empty(2 * nbytes, dtype=torch.uint8, device=runtime.device)
            runtime.all_gather(inp, out01, g01)
        if rank in (0, 2):
            inp2 = _ag_pattern(rank, 5000 + it, nbytes, runtime.device)
            out02 = torch.empty(2 * nbytes, dtype=torch.uint8, device=runtime.device)
            runtime.all_gather(inp2, out02, g02)

        runtime.stream.synchronize()

        if rank in (0, 1):
            for seg, peer in enumerate(g01.ranks):
                exp = _ag_pattern(peer, 1000 + it, nbytes, runtime.device)
                got = out01[seg * nbytes : (seg + 1) * nbytes]
                if not torch.equal(got, exp):
                    raise AssertionError(
                        f"#2 g01 corruption iter={it} seg={seg} peer={peer} "
                        f"(comm_buf reused before peer finished reading?)"
                    )
        if rank in (0, 2):
            for seg, peer in enumerate(g02.ranks):
                exp = _ag_pattern(peer, 5000 + it, nbytes, runtime.device)
                got = out02[seg * nbytes : (seg + 1) * nbytes]
                if not torch.equal(got, exp):
                    raise AssertionError(
                        f"#2 g02 corruption iter={it} seg={seg} peer={peer}"
                    )
        dist.barrier(group=runtime.world_pg)

    if g01 is not None:
        runtime.unregister_group(g01)
    if g02 is not None:
        runtime.unregister_group(g02)


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()
    assert 2 <= world <= 8, f"expects nproc in [2, 8], got {world}"

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=8 * 1024 * 1024,
        use_tma=False,
        use_copy_engine=False,
        enable_fused_path=True,           # default path for the #2 check
        pipeline_chunks=4,
        max_pipeline_chunks=4,
        pipeline_min_chunk_bytes=1024,    # let small slices pipeline in tests
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g_world = runtime.register_group(tuple(range(world)))

    _check_pipelined_unaligned(runtime, g_world)
    _check_arg_validation(runtime, g_world)
    _check_autotune_guards(runtime, g_world)
    _check_autotune_fused_ok(runtime, g_world)
    _check_fused_all2all_multichunk(runtime, g_world)
    _check_autotune_load_validation(device)

    if world >= 3:
        _check_fused_overlapping_groups(runtime)
    else:
        rank_print("skip #2 fused-overlap check (needs world>=3)")

    rank_print(f"ok review fixes world={world}")

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
