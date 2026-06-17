"""Phase-4 barrier: 1000 bare barriers across the group [0, 1].

Verifies:
  * the barrier kernel doesn't deadlock when run many times in tight loop;
  * epoch counter increments by 1 each call (host-side state);
  * tokens for distinct epochs are pairwise distinct.

Run:
    torchrun --standalone --nproc_per_node=2 tests/test_barrier_repeated.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
    compute_token,
)

from tests._harness import rank_print, setup_nccl, teardown  # noqa: E402


def main() -> int:
    device = setup_nccl()
    assert dist.get_world_size() == 2, "test_barrier_repeated expects nproc=2"

    config = SymmetricCollectiveConfig(
        max_group_size=2,
        max_collective_bytes=1 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)
    g = runtime.register_group((0, 1))

    n_iters = 1000
    expected_tokens = [
        compute_token(runtime.session_nonce, g.group_id, e)
        for e in range(n_iters)
    ]
    assert len(set(expected_tokens)) == n_iters, "tokens must be pairwise distinct"

    # Sanity: at start the epoch counter for g is 0.
    assert runtime._epochs.peek(g.group_id) == 0

    t0 = time.perf_counter()
    for i in range(n_iters):
        runtime.barrier(g)
    runtime.stream.synchronize()
    t1 = time.perf_counter()

    # After n_iters bare barriers, the epoch counter must be exactly n_iters.
    assert runtime._epochs.peek(g.group_id) == n_iters, (
        f"epoch peek = {runtime._epochs.peek(g.group_id)} != {n_iters}"
    )

    rank_print(
        f"ok {n_iters} bare barriers in {(t1 - t0) * 1000:.1f} ms "
        f"({(t1 - t0) / n_iters * 1e6:.2f} us/iter, host-only timing)"
    )

    # After all barriers, the per-edge double-buffered cells must hold the
    # last two tokens — slot ((n_iters - 1) & 1) has the most recent token
    # and the other slot has the token from the previous barrier. The
    # signal grid is laid out as u64[2 slots, world_size src].
    final_token = expected_tokens[-1]
    prev_token = expected_tokens[-2]
    last_slot = (n_iters - 1) & 1
    prev_slot = 1 - last_slot
    cells = runtime.regions.signal_buf.tensor.view(
        config.num_signal_slots, runtime.world_size
    )
    peer = 1 - runtime.rank  # 2-rank group {0,1}; peer is the other rank
    last_cell = int(cells[last_slot, peer].item())
    prev_cell = int(cells[prev_slot, peer].item())
    assert last_cell == final_token, (
        f"rank {runtime.rank} cell[slot={last_slot}, src={peer}] = "
        f"{last_cell:#x}, expected final token {final_token:#x}"
    )
    assert prev_cell == prev_token, (
        f"rank {runtime.rank} cell[slot={prev_slot}, src={peer}] = "
        f"{prev_cell:#x}, expected prev token {prev_token:#x}"
    )

    # The local edge_seq counter should have advanced exactly n_iters
    # times for the peer entry.
    seq_peer = int(runtime.edge_seq[peer].item())
    assert seq_peer == n_iters, (
        f"rank {runtime.rank} edge_seq[{peer}] = {seq_peer}, expected {n_iters}"
    )

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
