"""High-intensity barrier stress over random rank-mask combinations.

Rank 0 generates a deterministic random sequence of N mask integers, each in
``[1, 2**world)``. Each mask's set bits enumerate the participating ranks
(bit ``i`` → rank ``i``). The mask sequence is broadcast to every rank, so
all ranks agree on the schedule. Each rank then loops the schedule, calling
``barrier()`` for every mask whose bit it occupies, and skipping the rest.

This is the same shape as the fixed-schedule stress test it replaced, but
it scales trivially to any ``world`` (4, 8, …) without enumerating groups
by hand, and gives different group coverage each run (seedable for
reproducibility).

Hits exactly the failure mode the per-edge double-buffer redesign fixes:

  * frequent back-to-back hits on the same pair under different group
    masks would, under the old single-cell signaling, let rank r's
    finish/ack for barrier B overwrite r's finish/ack for the previous
    barrier A before the peer had read it — deadlocking the peer;
  * with the new per-edge double-buffered protocol, slots rotate per pair
    so consecutive barriers between the same pair land on different cells.

Verifies:
  * the host loop completes within the NCCL bootstrap timeout (no deadlock);
  * each rank's local ``edge_seq[peer]`` counter advanced exactly the
    number of times that ``(self, peer)`` co-occurred in the broadcast
    schedule — i.e. the GPU-side slot bookkeeping is self-consistent.

Run:
    torchrun --standalone --nproc_per_node=4 tests/test_barrier_stress.py
    torchrun --standalone --nproc_per_node=8 tests/test_barrier_stress.py

Optional env:
    GFC_STRESS_ITERS   number of mask draws (default 1024)
    GFC_STRESS_SEED    PRNG seed used on rank 0 (default 0xC0FFEE)
"""

from __future__ import annotations

import os
import random
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfc import (  # noqa: E402
    SymmetricCollectiveConfig,
    SymmetricCollectiveRuntime,
)

from tests._harness import rank_print, setup_nccl, teardown  # noqa: E402


def _mask_to_ranks(mask: int, world: int) -> tuple[int, ...]:
    return tuple(r for r in range(world) if mask & (1 << r))


def _broadcast_schedule(world: int, n: int, seed: int) -> list[int]:
    """Rank 0 picks N random non-zero masks in [1, 2**world); broadcast list."""
    if dist.get_rank() == 0:
        rng = random.Random(seed)
        masks = [rng.randrange(1, 1 << world) for _ in range(n)]
    else:
        masks = [0] * n
    obj_list: list = [masks]
    dist.broadcast_object_list(obj_list, src=0)
    out = list(obj_list[0])
    assert len(out) == n
    assert all(1 <= m < (1 << world) for m in out), "bad mask in broadcast schedule"
    return out


def _expected_edge_advance(rank: int, world: int, schedule: list[int]) -> list[int]:
    """For one walk of ``schedule``, how many times does each peer's
    ``edge_seq`` cell advance on ``rank``?

    The kernel grid is ``(group.size,)`` and every program advances the
    local ``edge_seq[peer]`` once — including the pid where ``peer == self``.
    So for every mask containing ``rank``, every set bit ``q`` of that mask
    contributes +1 to ``rank``'s counter for peer ``q``.
    """
    per_peer = [0] * world
    self_bit = 1 << rank
    for mask in schedule:
        if not (mask & self_bit):
            continue
        for q in range(world):
            if mask & (1 << q):
                per_peer[q] += 1
    return per_peer


def main() -> int:
    device = setup_nccl()
    world = dist.get_world_size()
    assert 2 <= world <= 16, f"test_barrier_stress expects 2<=nproc<=16, got {world}"

    n_iters = int(os.environ.get("GFC_STRESS_ITERS", "1024"))
    seed = int(os.environ.get("GFC_STRESS_SEED", str(0xC0FFEE)))
    assert n_iters > 0

    schedule = _broadcast_schedule(world, n_iters, seed)

    # Distinct masks actually present in the schedule — we register a
    # GroupDescriptor per unique mask. Group count is unbounded; each group
    # owns its own device rank list.
    unique_masks = sorted(set(schedule))

    config = SymmetricCollectiveConfig(
        max_group_size=world,
        max_collective_bytes=1 * 1024 * 1024,
    )
    runtime = SymmetricCollectiveRuntime(config, device=device)

    # Register every group once and build a mask -> descriptor lookup.
    mask_to_group = {}
    for mask in unique_masks:
        ranks = _mask_to_ranks(mask, world)
        mask_to_group[mask] = runtime.register_group(ranks)

    # Count this rank's member calls for nicer logging.
    self_bit = 1 << runtime.rank
    member_calls = sum(1 for m in schedule if m & self_bit)

    if runtime.rank == 0:
        rank_print(
            f"stress: world={world} seed={seed:#x} n_iters={n_iters} "
            f"unique_masks={len(unique_masks)}"
        )

    t0 = time.perf_counter()
    for mask in schedule:
        if not (mask & self_bit):
            continue
        runtime.barrier(mask_to_group[mask])
    runtime.stream.synchronize()
    t1 = time.perf_counter()

    # Verify edge_seq[peer] advanced exactly the expected count.
    expected = _expected_edge_advance(runtime.rank, world, schedule)
    seq_host = runtime.edge_seq.cpu().tolist()
    for peer in range(world):
        want = expected[peer]
        got = int(seq_host[peer])
        assert got == want, (
            f"rank {runtime.rank} edge_seq[{peer}] = {got}, expected {want}"
        )

    # Sanity: each registered group's host-side epoch counter equals the
    # number of times its mask appeared in the schedule.
    for mask, group in mask_to_group.items():
        if group.local_index < 0:
            continue
        scheduled = sum(1 for m in schedule if m == mask)
        peek = runtime._epochs.peek(group.group_id)
        assert peek == scheduled, (
            f"rank {runtime.rank} group {group.ranks} epoch peek = {peek}, "
            f"expected {scheduled}"
        )

    dist.barrier(group=runtime.world_pg)
    rank_print(
        f"ok stress: world={world} {member_calls} barrier calls "
        f"in {(t1 - t0) * 1000:.1f} ms "
        f"({(t1 - t0) / max(member_calls, 1) * 1e6:.2f} us/call avg) "
        f"across {len(unique_masks)} unique groups"
    )

    runtime.shutdown()
    teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
