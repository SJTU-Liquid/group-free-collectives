"""SymmetricCollectiveRuntime.

Phase 2 scope: bootstrap (init_process_group already done by caller),
``enable_symm_mem_for_group`` on the world PG, allocation of the four
symmetric regions, footprint logging, session-nonce broadcast, group pool
table on device. Collective methods (``barrier``, ``all_gather``, ``all2all``,
``p2p_put``, ``p2p_get``) are filled in by later phases.
"""

from __future__ import annotations

import os
import secrets
import threading
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from gfc.autotune import AutotuneTable, Policy
from gfc.buffer import SymmRegions, allocate_symm_regions, zero_signal_grids
from gfc.config import SymmetricCollectiveConfig, _require
from gfc.env import read_env_overrides
from gfc.group import GroupDescriptor, _EpochManager, compute_token, stable_hash64
from gfc.kernels._common import vec_width_bytes
from gfc.kernels.all2all import launch_all2all_pull
from gfc.kernels.all_gather import launch_all_gather_pull
from gfc.kernels.barrier import launch_barrier
from gfc.kernels.copy_engine import (
    launch_all2all_copy_engine,
    launch_all_gather_copy_engine,
    launch_p2p_copy_engine,
)
from gfc.kernels.copy_pull import launch_pull_copy
from gfc.kernels.fused import launch_all2all_fused, launch_all_gather_fused
from gfc.kernels.p2p_put import launch_p2p_put_copy
from gfc.kernels.tma_paths import (
    check_collective_tma_gates,
    launch_all2all_tma,
    launch_all_gather_tma,
    launch_p2p_copy_tma,
)
from gfc.tma_probe import TMAUnsupportedError, probe_tma_supported
from gfc.logging import get_logger, log_footprint


_NONCE_STORE_KEY = "gfc/session_nonce"


class SymmetricCollectiveRuntime:
    """Single-stream runtime over a one-time world-group symmetric rendezvous.

    Owns one CUDA stream (``self.stream``); all collective submissions are
    serialized through ``self._submit_lock`` and enqueued onto it. The
    bootstrap process group is used only for rendezvous, the session-nonce
    broadcast, and debug-mode consistency checks — never on a collective
    data path.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        config: SymmetricCollectiveConfig,
        device: torch.device,
        world_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        env = read_env_overrides()
        if env.debug:
            config = _replace(config, enable_debug_checks=True)
        if env.use_tma:
            config = _replace(config, use_tma=True)
        if env.use_copy_engine:
            config = _replace(config, use_copy_engine=True)
        if env.copy_sms is not None:
            config = _replace(config, copy_sms=env.copy_sms)
        if env.timeout_ms is not None:
            config = _replace(config, timeout_ms=env.timeout_ms)
        if env.max_pipeline_chunks is not None:
            config = _replace(config, max_pipeline_chunks=env.max_pipeline_chunks)
        if env.pipeline_chunks is not None:
            config = _replace(config, pipeline_chunks=env.pipeline_chunks)
        if env.enable_fused_path is not None:
            config = _replace(config, enable_fused_path=env.enable_fused_path)
        if env.fused_num_channels is not None:
            config = _replace(config, fused_num_channels=env.fused_num_channels)
        if env.fused_chunk_size is not None:
            config = _replace(config, fused_chunk_size=env.fused_chunk_size)
        if env.autotune_config_path is not None:
            config = _replace(config, autotune_config_path=env.autotune_config_path)

        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"gfc requires a CUDA device, got {self.device!r}")

        _require(
            dist.is_initialized(),
            "torch.distributed must be initialised before SymmetricCollectiveRuntime",
        )
        self.world_pg = world_group if world_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.world_pg)
        self.world_size = dist.get_world_size(self.world_pg)

        # The data-plane stream. All collectives are enqueued here.
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        # Auxiliary stream used by the pipelined-staging path to overlap the
        # local input-> comm_buf copy of chunk k+1 with the cross-rank pull
        # of chunk k on ``self.stream``. Data buffer reuse is still ordered by
        # the main stream and the post barrier.
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._submit_lock = threading.Lock()

        # Autotune table: when present, every all_gather / all2all call
        # consults this for path + knob selection. Loading is once-only.
        self.autotune: AutotuneTable | None = None
        if config.autotune_config_path:
            self.autotune = AutotuneTable.from_json(config.autotune_config_path)
            # Fail fast: a fused rule may name more channels than this runtime
            # allocated ``step_pad`` for (the symmetric grid is sized once from
            # ``config.fused_num_channels`` and rendezvoused at that size). The
            # dispatch guard would otherwise only reject this once a payload
            # large enough to select that rule arrives — a latent failure deep
            # in a run. Surface it here with the exact knob to provision.
            need_channels = self.autotune.max_fused_num_channels()
            if need_channels > config.fused_num_channels:
                raise ValueError(
                    f"autotune config {config.autotune_config_path!r} contains a "
                    f"fused policy requesting fused_num_channels={need_channels}, "
                    f"but this runtime is provisioned for "
                    f"config.fused_num_channels={config.fused_num_channels}. The "
                    f"symmetric step_pad grid is sized once from the config and "
                    f"cannot grow per call. Provision the runtime to match before "
                    f"loading this table, e.g. set "
                    f"SYMM_COLL_FUSED_NUM_CHANNELS={need_channels} (or pass "
                    f"fused_num_channels={need_channels} in "
                    f"SymmetricCollectiveConfig)."
                )

        # Persistent-kernel grid size matches the device SM count so each
        # SM owns one program that loops over (peer, tile) work items.
        self.num_sms = int(
            torch.cuda.get_device_properties(self.device).multi_processor_count
        )
        if config.enable_fused_path:
            self._check_fused_launch_capacity(config.fused_num_channels)
        self.copy_sms = min(self.num_sms, int(config.copy_sms))
        self.copy_engine_enabled = bool(config.use_copy_engine)

        # Spin-wait watchdog bound for the barrier / fused-receiver kernels
        # (0 disables). A configured timeout turns an otherwise-unbounded
        # acquire spin — mismatched barrier order, a dead peer, a lost token —
        # into a kernel ``trap`` (a diagnosable CUDA error on the next host
        # sync) instead of a GPU that hangs forever.
        self._timeout_ns = int(config.timeout_ms) * 1_000_000

        # Symmetric-memory enable + region allocation must run on every rank
        # in the world group concurrently. enable_symm_mem_for_group is a
        # collective; rendezvous is too.
        enable_symm_mem_for_group(self.world_pg.group_name)
        self.regions: SymmRegions = allocate_symm_regions(
            config, self.device, self.world_pg, self.world_size
        )

        # Per-rank local edge-sequence counter, indexed by *global* peer rank.
        # Drives the per-edge double-buffer slot inside the barrier kernel:
        # ``slot = edge_seq[peer] & 1``. Local-only; peers do not read it,
        # so it is not in symmetric memory. Lives on the device so slot
        # selection stays on the runtime stream; host never participates in
        # slot selection.
        self.edge_seq = torch.zeros(
            self.world_size, dtype=torch.uint64, device=self.device
        )

        # Session nonce: rank 0 generates, broadcast via TCPStore-backed
        # bootstrap PG using object collectives.
        self.session_nonce = self._broadcast_session_nonce()
        _require(self.session_nonce != 0, "session_nonce must be non-zero")

        # Group registry and per-group epoch manager. Each registered group
        # owns its own device-resident ``uint32[size]`` rank list, allocated in
        # ``register_group`` and freed in ``unregister_group``. There is no
        # preallocated pool and no cap on the number of live groups.
        self._registered: dict[int, GroupDescriptor] = {}
        self._epochs = _EpochManager()

        # Optional TMA path. If use_tma is set, probe must pass — no silent
        # fallback. If it fails we raise; users can flip use_tma=False to
        # keep going on the vec path.
        self.tma_enabled = False
        if config.use_tma and self.copy_engine_enabled:
            raise ValueError("use_tma and use_copy_engine are mutually exclusive")

        if config.use_tma:
            if probe_tma_supported(self):
                self.tma_enabled = True
                get_logger().info(
                    "TMA enabled (peer-pointer load verified)"
                )
            else:
                raise TMAUnsupportedError(
                    "TMA probe failed on this hardware/driver; "
                    "set use_tma=False to use the vec fallback"
                )
        else:
            get_logger().info("TMA disabled (use_tma=False)")
        if self.copy_engine_enabled:
            get_logger().info("copy-engine path enabled (cudaMemcpyAsync)")

        log = get_logger()
        if self.rank == 0:
            log_footprint(
                max_collective_bytes=config.max_collective_bytes,
                num_signal_slots=config.num_signal_slots,
                world_size=self.world_size,
                enable_debug_checks=config.enable_debug_checks,
                fused_num_channels=config.fused_num_channels,
                fused_max_chunks_per_channel=config.fused_max_chunks_per_channel,
            )
        log.info(
            "rank=%d world=%d device=%s symm_mem enabled (group=%s)",
            self.rank,
            self.world_size,
            self.device,
            self.world_pg.group_name,
        )

    # ----------------------------------------------------------------- nonce
    def _broadcast_session_nonce(self) -> int:
        """Rank 0 picks a non-zero u64 nonce; broadcast over bootstrap PG."""
        if self.rank == 0:
            nonce = secrets.randbits(64) | 1
        else:
            nonce = 0
        obj_list = [nonce]
        dist.broadcast_object_list(obj_list, src=0, group=self.world_pg)
        n = int(obj_list[0])
        if self.config.enable_debug_checks:
            # Cross-check that everyone agrees on the broadcast value.
            gathered: list[Optional[int]] = [None] * self.world_size
            dist.all_gather_object(gathered, n, group=self.world_pg)
            _require(
                all(g == n for g in gathered),
                f"session_nonce mismatch across ranks: {gathered}",
            )
        return n

    # ------------------------------------------------- group registration
    def register_group(
        self,
        ranks: Sequence[int],
        group_id: Optional[int] = None,
    ) -> GroupDescriptor:
        rt = tuple(int(r) for r in ranks)
        _require(
            0 < len(rt) <= self.config.max_group_size,
            f"|group|={len(rt)} must be in (0, {self.config.max_group_size}]",
        )
        _require(len(set(rt)) == len(rt), f"ranks must be unique: {rt}")
        _require(
            all(0 <= r < self.world_size for r in rt),
            f"ranks out of [0,{self.world_size}): {rt}",
        )

        gid = (
            group_id
            if group_id is not None
            else stable_hash64(b"gfc-group-v1", rt)
        )

        # Registry mutations are serialized through the same lock collective
        # submission uses, so register/unregister and an in-flight collective
        # never race on ``self._registered`` (the debug consistency check is a
        # blocking collective, so it runs *outside* the lock).
        with self._submit_lock:
            existing = self._registered.get(gid)
            if existing is not None:
                # Idempotent re-registration is only safe when the rank set
                # matches. An explicit ``group_id`` collision (or a 64-bit hash
                # collision) with a *different* rank set would otherwise hand
                # back a descriptor whose ``ranks_dev`` drives kernels at the
                # wrong peers — a silent deadlock or cross-talk. Fail loudly.
                _require(
                    existing.ranks == rt,
                    f"group_id {gid:#x} already registered with ranks "
                    f"{existing.ranks}; refusing to reuse it for {rt}",
                )
                return existing

            local_index = rt.index(self.rank) if self.rank in rt else -1
            # Only members ever launch a kernel that reads the rank list, so a
            # non-member descriptor carries no device tensor.
            ranks_dev = self._alloc_group_ranks(rt) if local_index >= 0 else None
            desc = GroupDescriptor(
                group_id=gid,
                ranks=rt,
                local_index=local_index,
                size=len(rt),
                ranks_dev=ranks_dev,
            )
            self._registered[gid] = desc

        # The consistency cross-check is a *world* ``all_gather_object``, so
        # every world rank must call it. Only members call ``register_group``
        # for a genuine subgroup (no per-subgroup process groups exist by
        # design), so running the world collective for a subgroup would hang
        # the members on the absent non-members. Restrict the check to the
        # full-world group, where membership == the whole world and every rank
        # therefore participates. Subgroup definitions can't be cross-checked
        # this way without a subgroup PG, which the design forbids.
        if self.config.enable_debug_checks and len(rt) == self.world_size:
            self._dbg_verify_group_consistency(gid, rt)

        return desc

    def _alloc_group_ranks(self, ranks: tuple[int, ...]) -> torch.Tensor:
        """Allocate a device-resident ``uint32[len(ranks)]`` ordered rank list.

        Sized exactly to the group: every kernel reads only the first
        ``group_size`` entries (the fused path masks lanes ``>= group_size``),
        so no sentinel padding is needed.

        Allocated on ``self.stream`` so the host->device fill is FIFO-ordered
        before the barrier/copy kernels that later read it on the same stream.
        This makes the ordering explicit instead of relying on the incidental
        host-synchronous behaviour of a small pageable copy.
        """
        with torch.cuda.stream(self.stream):
            return torch.tensor(ranks, dtype=torch.uint32, device=self.device)

    def unregister_group(self, group: GroupDescriptor) -> None:
        """Release a group registered with :meth:`register_group`.

        Drops the runtime's registry entry and frees the group's device rank
        list. Idempotent: releasing an unknown or already-released group is a
        no-op. Releasing a *stale* handle whose ``group_id`` has since been
        re-registered to a different descriptor is also a no-op — only the
        descriptor the registry currently holds is ever evicted, so a stale
        handle can never free the live group's rank list.

        The per-group epoch counter is intentionally **not** dropped: barrier
        tokens must stay strictly monotonic for a ``group_id`` across the whole
        session (the per-edge double buffer relies on never reusing a token
        within its reuse window), so a re-registered group resumes its epoch
        sequence rather than restarting at 0.

        The rank-list tensor is ``record_stream``-d against the submission
        stream before its reference is dropped, so the caching allocator defers
        reclaiming the block until collectives already queued on ``self.stream``
        (the only stream that reads the rank list) drain past this point — a
        non-blocking free with no use-after-free of a row a queued barrier/copy
        still reads. Registration is a control-plane operation, not part of the
        collective hot path.
        """
        with self._submit_lock:
            # Evict only if the registry holds *this exact* handle. Popping by
            # id would let a stale handle free a re-registered live group.
            if self._registered.get(group.group_id) is not group:
                return
            del self._registered[group.group_id]
            ranks_dev = group.ranks_dev
            if ranks_dev is not None:
                ranks_dev.record_stream(self.stream)
                # Frozen dataclass: clear the backing reference so the block is
                # released even though the caller still holds the handle.
                object.__setattr__(group, "ranks_dev", None)

    def _dbg_verify_group_consistency(
        self, gid: int, ranks: tuple[int, ...]
    ) -> None:
        gathered: list[Optional[tuple[int, tuple[int, ...]]]] = [None] * self.world_size
        dist.all_gather_object(gathered, (gid, ranks), group=self.world_pg)
        ref = (gid, ranks)
        for r, g in enumerate(gathered):
            if g != ref:
                raise RuntimeError(
                    f"register_group consistency check failed: "
                    f"rank {r} sees {g}, local sees {ref}"
                )

    # ----------------------------------------------------------- group internals
    def _group_ranks_row_ptr(self, desc: GroupDescriptor) -> int:
        """Device address of ``desc``'s ordered rank-list tensor."""
        _require(
            desc.ranks_dev is not None,
            "group has no device rank list (non-member or already unregistered)",
        )
        return int(desc.ranks_dev.data_ptr())

    def _check_fused_launch_capacity(self, num_channels: int) -> None:
        _require(num_channels > 0, "fused_num_channels must be > 0")
        required_ctas = 2 * int(num_channels)
        _require(
            required_ctas <= self.num_sms,
            f"fused path requires 2*fused_num_channels={required_ctas} CTAs "
            f"to co-reside, but this device has {self.num_sms} SMs; lower "
            "fused_num_channels or disable the fused path",
        )

    def _preflight_tma_collective(
        self, *, slice_bytes: int, comm_buf_offset: int, dst_ptr: int
    ) -> None:
        check_collective_tma_gates(
            slice_bytes=slice_bytes,
            comm_buf_offset=comm_buf_offset,
            dst_ptr=dst_ptr,
        )

    def _barrier_on(self, group: GroupDescriptor, row_ptr: int, token: int) -> None:
        """Launch one per-edge barrier for ``group`` on the current stream.

        Must be called inside ``with torch.cuda.stream(self.stream):``. Centralises
        the barrier-kernel boilerplate (single source of truth for the launch
        contract) and carries the configured ``timeout_ns`` so the in-kernel
        watchdog bounds the wait.
        """
        launch_barrier(
            token=token,
            self_global_rank=self.rank,
            group_ranks_row_ptr=row_ptr,
            signal_ptrs_dev=self.signal_buf_ptrs_dev,
            edge_seq_ptr=self.edge_seq_ptr,
            group_size=group.size,
            world_size=self.world_size,
            timeout_ns=self._timeout_ns,
        )

    # --------------------------------------------------------- stream handoff
    def _ingest_user_tensors(self, *tensors: torch.Tensor) -> None:
        """Hand caller-owned tensors safely onto the runtime's private streams.

        Every collective stages the caller's ``input``/``src`` into ``comm_buf``
        on ``self.stream`` (and ``self.copy_stream`` for the pipelined path),
        and writes the caller's ``output``/``dst`` there too. Those tensors were
        produced — and may be freed — on the caller's *current* stream. Without
        ordering, the staging copy can read an input before the op that produced
        it has run, or after the block has been recycled by the caching allocator
        (manifesting as stale/zero bytes on the fast path, masked only by
        JIT-compile delays on the very first call). So:

        * make the runtime streams wait for the caller's current stream, so
          inputs are fully produced before we read them, and
        * ``record_stream`` each tensor against the runtime streams, so the
          allocator defers freeing the block until the runtime is done with it.

        The caller is symmetrically responsible for ordering *its* stream after
        ``self.stream`` before reading an ``output`` (e.g. ``stream.synchronize``
        or :meth:`record_event` + the caller waiting on it).
        """
        current = torch.cuda.current_stream(self.device)
        if current != self.stream:
            self.stream.wait_stream(current)
        if current != self.copy_stream:
            self.copy_stream.wait_stream(current)
        for t in tensors:
            t.record_stream(self.stream)
            t.record_stream(self.copy_stream)

    # --------------------------------------------------------------- collectives
    def barrier(
        self,
        group: GroupDescriptor,
    ) -> None:
        """Bare release/acquire-sys barrier across ``group``.

        Consumes one epoch for ``group.group_id`` (``+= 1``). Synchronisation
        is per-edge double-buffered: for every peer ``p`` in ``group``, the
        kernel selects slot ``edge_seq[p] & 1`` and writes/reads that slot
        through ``signal_buf``.

        Ordering constraint — read this before driving GFC from your own
        scheduler
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Within any **rank pair** (p, q), the sequence of barriers that
        involve both p and q must be identical on the two endpoints. The
        protocol uses ``edge_seq[peer] & 1`` to pick the double-buffer slot
        and assumes both ends agree on which slot belongs to which barrier;
        if the two ranks issue barriers in different orders, slot indices
        diverge and the pair deadlocks (rank p waits in slot 0 while rank q
        writes slot 1, or two pair-involving barriers race for the same
        slot and one overwrites the other).

        Concretely: if rank 0 issues ``barrier(G_AB)`` then ``barrier(G_AC)``,
        every other rank that is in either group **must** see the same
        relative order of those two barriers when restricted to the pairs
        they participate in. A central single-threaded driver (one
        ``submit_lock`` per rank, broadcasting the same group schedule)
        trivially satisfies this. Per-rank independent decisions do not —
        if rank 1 and rank 2 might disagree on whether ``G_12`` or some
        other pair-involving group runs first, the protocol is unsafe.
        """
        _require(group.local_index >= 0, "this rank is not a member of the group")
        _require(
            group.size <= self.config.max_group_size,
            f"group.size={group.size} > max_group_size={self.config.max_group_size}",
        )

        with self._submit_lock:
            row_ptr = self._group_ranks_row_ptr(group)
            epoch = self._epochs.next_for_barrier(group.group_id)
            token = compute_token(self.session_nonce, group.group_id, epoch)
            with torch.cuda.stream(self.stream):
                self._barrier_on(group, row_ptr, token)

    # -------------------------------------------------------------- all_gather
    def all_gather(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        *,
        pipeline_chunks: Optional[int] = None,
    ) -> None:
        """Pull-model all_gather across ``group``.

        ``output`` must hold ``group.size * input_.nbytes`` bytes. After
        return, ``output[i * input_.nbytes : (i+1) * input_.nbytes]`` contains
        rank ``group.ranks[i]``'s input.

        ``pipeline_chunks`` overrides ``config.pipeline_chunks`` for this
        call. When effectively > 1 (and on the vec data path), the local
        stage of chunk ``k+1`` runs on ``copy_stream`` while the remote pull
        of chunk ``k`` runs on the main stream.
        """
        _validate_collective_args(
            input_, output, group, self.device, self.config,
            kind="all_gather",
        )
        _require(
            output.nbytes == group.size * input_.nbytes,
            f"output size {output.nbytes} != group.size * input.nbytes "
            f"{group.size * input_.nbytes} (exact: floor-division would let an "
            f"oversized output leave a tail untouched)",
        )

        slice_bytes = input_.nbytes
        chunks = self._effective_pipeline_chunks(slice_bytes, pipeline_chunks)
        policy = None
        if self.autotune is not None:
            policy = self.autotune.policy_for("allgather", group.size, slice_bytes)

        with self._submit_lock:
            self._ingest_user_tensors(input_, output)
            comm_buf_offset = 0
            row_ptr = self._group_ranks_row_ptr(group)
            sig_ptrs = self.signal_buf_ptrs_dev
            cb_ptrs = self.comm_buf_ptrs_dev

            if policy is not None:
                self._dispatch_all_gather(
                    policy=policy,
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                )
            elif self.config.enable_fused_path:
                self._check_fused_launch_capacity(self.config.fused_num_channels)
                # Fused single-kernel path. One pre-barrier (start line), then
                # a persistent kernel that stages + publishes step counters +
                # pulls all chunks, then a post-barrier. The step token proves
                # *this* rank pulled what it needs, but NOT that peers finished
                # pulling from this rank's comm_buf — so a following collective
                # on a different (overlapping) group could overwrite comm_buf
                # while a peer is still reading. The post-barrier closes that
                # reuse hazard exactly as the non-fused path does.
                pre_e = self._epochs.next_for_barrier(group.group_id)
                step_base_e = self._epochs.next_for_barrier(group.group_id)
                post_e = self._epochs.next_for_barrier(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                step_base = compute_token(
                    self.session_nonce, group.group_id, step_base_e
                )
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

                with torch.cuda.stream(self.stream):
                    self._barrier_on(group, row_ptr, pre_tok)
                    launch_all_gather_fused(
                        input_ptr=int(input_.data_ptr()),
                        output_ptr=int(output.data_ptr()),
                        comm_buf_ptrs_dev=cb_ptrs,
                        step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                        group_ranks_row_ptr=row_ptr,
                        comm_buf_offset=comm_buf_offset,
                        self_global_rank=self.rank,
                        self_local_index=group.local_index,
                        slice_bytes=slice_bytes,
                        group_size=group.size,
                        world_size=self.world_size,
                        max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                        num_channels=self.config.fused_num_channels,
                        chunk_bytes=self.config.fused_chunk_size,
                        step_base=step_base,
                        alignment=self.config.alignment,
                        timeout_ns=self._timeout_ns,
                    )
                    # Post-barrier: prove every peer finished pulling this
                    # rank's comm_buf before a later collective reuses it.
                    self._barrier_on(group, row_ptr, post_tok)
            elif chunks <= 1:
                if self.tma_enabled:
                    self._preflight_tma_collective(
                        slice_bytes=slice_bytes,
                        comm_buf_offset=comm_buf_offset,
                        dst_ptr=int(output.data_ptr()),
                    )
                pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

                with torch.cuda.stream(self.stream):
                    # 1. Stage local input into our own comm_buf.
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + slice_bytes
                    ]
                    local_buf.copy_(input_.view(torch.uint8).reshape(-1))

                    # 2. Pre-barrier: every peer must have its slice in place.
                    self._barrier_on(group, row_ptr, pre_tok)

                    # 3. Pull copy from each peer into the output.
                    if self.copy_engine_enabled:
                        launch_all_gather_copy_engine(
                            peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks=group.ranks,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled:
                        launch_all_gather_tma(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(
                            self.regions.comm_buf.local_ptr,
                            int(output.data_ptr()),
                            slice_bytes,
                        )
                        launch_all_gather_pull(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )

                    # 4. Post-barrier: subsequent collectives must wait for
                    #    these pulls to complete before reusing comm_buf.
                    self._barrier_on(group, row_ptr, post_tok)
            else:
                # Pipelined chunked path on the vec kernels. The helper issues
                # its own post-barrier, so there is no epoch/token to thread back.
                self._all_gather_pipelined(
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                    chunks=chunks,
                )

    # ----------------------------------------------------------------- all2all
    def all2all(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        *,
        slice_bytes: Optional[int] = None,
        pipeline_chunks: Optional[int] = None,
    ) -> None:
        """Pull-model all-to-all across ``group``.

        ``input_`` is laid out as ``group.size`` contiguous slices of equal
        ``slice_bytes`` bytes — slice ``t`` is destined for ``group.ranks[t]``.
        After return, ``output[i * slice_bytes : (i+1) * slice_bytes]`` holds
        slice ``self_local_index`` from peer ``group.ranks[i]``.

        ``pipeline_chunks`` overrides ``config.pipeline_chunks`` for this
        call. When effectively > 1 (and on the vec data path), the local
        stage of chunk ``k+1`` runs on ``copy_stream`` while the remote pull
        of chunk ``k`` runs on the main stream.
        """
        _validate_collective_args(
            input_, output, group, self.device, self.config, kind="all2all"
        )
        _require(
            input_.nbytes == output.nbytes,
            f"all2all: input.nbytes={input_.nbytes} output.nbytes={output.nbytes}",
        )
        if slice_bytes is None:
            _require(
                input_.nbytes % group.size == 0,
                f"all2all: input.nbytes={input_.nbytes} not divisible by "
                f"group.size={group.size}",
            )
            slice_bytes = input_.nbytes // group.size
        _require(slice_bytes > 0, "all2all: slice_bytes must be > 0")
        # Exact coverage: input/output must be precisely ``group.size`` slices.
        # With an explicit ``slice_bytes`` the capacity check below only bounds
        # the high end; without this an over/under-sized buffer would silently
        # leave a tail untouched (or fault the staging copy).
        _require(
            input_.nbytes == slice_bytes * group.size,
            f"all2all: input.nbytes={input_.nbytes} != slice_bytes*group.size="
            f"{slice_bytes * group.size}",
        )
        _require(
            slice_bytes * group.size <= self.config.max_collective_bytes,
            f"all2all: slice_bytes*group.size = {slice_bytes*group.size} > "
            f"max_collective_bytes {self.config.max_collective_bytes}",
        )

        chunks = self._effective_pipeline_chunks(slice_bytes, pipeline_chunks)
        policy = None
        if self.autotune is not None:
            policy = self.autotune.policy_for("all2all", group.size, slice_bytes)

        with self._submit_lock:
            self._ingest_user_tensors(input_, output)
            comm_buf_offset = 0
            row_ptr = self._group_ranks_row_ptr(group)
            sig_ptrs = self.signal_buf_ptrs_dev
            cb_ptrs = self.comm_buf_ptrs_dev
            staged_bytes = slice_bytes * group.size

            if policy is not None:
                self._dispatch_all2all(
                    policy=policy,
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                    staged_bytes=staged_bytes,
                )
            elif self.config.enable_fused_path:
                self._check_fused_launch_capacity(self.config.fused_num_channels)
                # Fused single-kernel all2all. See ``all_gather`` for the
                # protocol overview (incl. why a post-barrier is required);
                # the all2all variant partitions along the per-peer-slice axis.
                pre_e = self._epochs.next_for_barrier(group.group_id)
                step_base_e = self._epochs.next_for_barrier(group.group_id)
                post_e = self._epochs.next_for_barrier(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                step_base = compute_token(
                    self.session_nonce, group.group_id, step_base_e
                )
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

                with torch.cuda.stream(self.stream):
                    self._barrier_on(group, row_ptr, pre_tok)
                    launch_all2all_fused(
                        input_ptr=int(input_.data_ptr()),
                        output_ptr=int(output.data_ptr()),
                        comm_buf_ptrs_dev=cb_ptrs,
                        step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                        group_ranks_row_ptr=row_ptr,
                        comm_buf_offset=comm_buf_offset,
                        self_global_rank=self.rank,
                        self_local_index=group.local_index,
                        slice_bytes=slice_bytes,
                        group_size=group.size,
                        world_size=self.world_size,
                        max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                        num_channels=self.config.fused_num_channels,
                        chunk_bytes=self.config.fused_chunk_size,
                        step_base=step_base,
                        alignment=self.config.alignment,
                        timeout_ns=self._timeout_ns,
                    )
                    # Post-barrier: prove every peer finished pulling this
                    # rank's comm_buf before a later collective reuses it.
                    self._barrier_on(group, row_ptr, post_tok)
            elif chunks <= 1:
                if self.tma_enabled:
                    self._preflight_tma_collective(
                        slice_bytes=slice_bytes,
                        comm_buf_offset=comm_buf_offset,
                        dst_ptr=int(output.data_ptr()),
                    )
                pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
                pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
                post_tok = compute_token(self.session_nonce, group.group_id, post_e)

                with torch.cuda.stream(self.stream):
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + staged_bytes
                    ]
                    local_buf.copy_(input_.view(torch.uint8).reshape(-1)[:staged_bytes])

                    self._barrier_on(group, row_ptr, pre_tok)

                    if self.copy_engine_enabled:
                        launch_all2all_copy_engine(
                            peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks=group.ranks,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled:
                        launch_all2all_tma(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(
                            self.regions.comm_buf.local_ptr,
                            int(output.data_ptr()),
                            slice_bytes,
                        )
                        launch_all2all_pull(
                            comm_buf_ptrs_dev=cb_ptrs,
                            dst_ptr=int(output.data_ptr()),
                            group_ranks_row_ptr=row_ptr,
                            comm_buf_offset=comm_buf_offset,
                            self_local_index=group.local_index,
                            slice_bytes=slice_bytes,
                            group_size=group.size,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )

                    self._barrier_on(group, row_ptr, post_tok)
            else:
                # Pipelined chunked path issues its own post-barrier internally.
                self._all2all_pipelined(
                    input_=input_,
                    output=output,
                    group=group,
                    comm_buf_offset=comm_buf_offset,
                    row_ptr=row_ptr,
                    sig_ptrs=sig_ptrs,
                    cb_ptrs=cb_ptrs,
                    slice_bytes=slice_bytes,
                    chunks=chunks,
                )

    # ----------------------------------------------------------------- p2p
    def p2p_put(
        self,
        dst_rank: int,
        src: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Push protocol — sender side.

        Stages ``src`` into local ``comm_buf``, runs the pre-barrier
        on the 2-rank ordered group ``[self.rank, dst_rank]``, issues a remote
        copy from local comm_buf into ``dst_rank``'s comm_buf, then the
        post-barrier. The partner rank must call
        :meth:`p2p_put_recv` with mirrored args on the same submission ordinal.
        """
        _require(dst_rank != self.rank, "p2p_put: dst_rank must differ from self.rank")
        self._p2p_protocol(
            src_rank=self.rank,
            dst_rank=dst_rank,
            local_tensor=src,
            role="sender",
            protocol="put",
            nbytes=nbytes,
        )

    def p2p_put_recv(
        self,
        src_rank: int,
        dst: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Push protocol — receiver side. Participates in the 2-rank barriers
        on ``[src_rank, self.rank]`` and copies the staged comm_buf into
        ``dst`` after the post-barrier."""
        _require(src_rank != self.rank, "p2p_put_recv: src_rank must differ from self.rank")
        self._p2p_protocol(
            src_rank=src_rank,
            dst_rank=self.rank,
            local_tensor=dst,
            role="receiver",
            protocol="put",
            nbytes=nbytes,
        )

    def p2p_get(
        self,
        src_rank: int,
        dst: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Pull protocol — receiver side. Issues pre-barrier on
        ``[src_rank, self.rank]``, ``pull_copy_kernel`` from the peer's
        ``comm_buf`` into ``dst``, then post-barrier. The partner
        rank must call :meth:`p2p_get_serve` to stage and participate."""
        _require(src_rank != self.rank, "p2p_get: src_rank must differ from self.rank")
        self._p2p_protocol(
            src_rank=src_rank,
            dst_rank=self.rank,
            local_tensor=dst,
            role="receiver",
            protocol="get",
            nbytes=nbytes,
        )

    def p2p_get_serve(
        self,
        dst_rank: int,
        src: torch.Tensor,
        *,
        nbytes: Optional[int] = None,
    ) -> None:
        """Pull protocol — sender side. Stages ``src`` into local
        ``comm_buf`` and participates in the 2-rank barriers so
        ``dst_rank``'s :meth:`p2p_get` can pull from it."""
        _require(dst_rank != self.rank, "p2p_get_serve: dst_rank must differ from self.rank")
        self._p2p_protocol(
            src_rank=self.rank,
            dst_rank=dst_rank,
            local_tensor=src,
            role="sender",
            protocol="get",
            nbytes=nbytes,
        )

    # ------------------------------------------------------ p2p shared core
    def _p2p_protocol(
        self,
        *,
        src_rank: int,
        dst_rank: int,
        local_tensor: torch.Tensor,
        role: str,
        protocol: str,
        nbytes: Optional[int],
    ) -> None:
        _require(
            local_tensor.is_cuda and local_tensor.device == self.device,
            f"p2p: tensor must be on runtime device {self.device}, got {local_tensor.device}",
        )
        _require(local_tensor.is_contiguous(), "p2p: tensor must be contiguous")
        if nbytes is None:
            nbytes = local_tensor.nbytes
        _require(
            0 < nbytes <= local_tensor.nbytes,
            f"p2p: nbytes={nbytes} must be in (0, {local_tensor.nbytes}]",
        )
        _require(
            nbytes <= self.config.max_collective_bytes,
            f"p2p: nbytes={nbytes} > max_collective_bytes={self.config.max_collective_bytes}",
        )

        # p2p groups are registered on demand and intentionally cached for the
        # session: register_group is idempotent, so a repeated (src, dst) pair
        # reuses its descriptor (and its monotonic epoch counter — required for
        # token uniqueness). The rank list is a tiny uint32[2]; the cache is
        # bounded by the number of distinct ordered pairs (<= world_size**2) and
        # is freed wholesale by shutdown(). Callers do not hold the descriptor,
        # so there is deliberately no per-call unregister.
        group = self.register_group((src_rank, dst_rank))
        _require(group.local_index >= 0, "this rank is not in the p2p group")

        with self._submit_lock:
            self._ingest_user_tensors(local_tensor)
            pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
            pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
            post_tok = compute_token(self.session_nonce, group.group_id, post_e)

            comm_buf_offset = 0
            row_ptr = self._group_ranks_row_ptr(group)
            sig_ptrs = self.signal_buf_ptrs_dev
            cb_local_ptr = self.regions.comm_buf.local_ptr + comm_buf_offset

            with torch.cuda.stream(self.stream):
                # Stage on the sender side.
                if role == "sender":
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + nbytes
                    ]
                    local_buf.copy_(local_tensor.view(torch.uint8).reshape(-1)[:nbytes])

                self._barrier_on(group, row_ptr, pre_tok)

                if protocol == "put" and role == "sender":
                    # Remote write: src is local comm_buf, dst is peer comm_buf.
                    peer_comm_ptrs = self.regions.comm_buf.handle.buffer_ptrs
                    dst_ptr_remote = int(peer_comm_ptrs[dst_rank]) + comm_buf_offset
                    if self.copy_engine_enabled:
                        launch_p2p_copy_engine(
                            src_ptr=cb_local_ptr,
                            dst_ptr=dst_ptr_remote,
                            nbytes=nbytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled and nbytes % 16 == 0:
                        launch_p2p_copy_tma(
                            src_ptr=cb_local_ptr,
                            dst_ptr=dst_ptr_remote,
                            nbytes=nbytes,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(cb_local_ptr, dst_ptr_remote, nbytes)
                        launch_p2p_put_copy(
                            src_ptr=cb_local_ptr,
                            dst_ptr=dst_ptr_remote,
                            nbytes=nbytes,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )
                elif protocol == "get" and role == "receiver":
                    # Remote read: src is peer comm_buf, dst is local user tensor.
                    peer_comm_ptrs = self.regions.comm_buf.handle.buffer_ptrs
                    src_ptr_remote = int(peer_comm_ptrs[src_rank]) + comm_buf_offset
                    dst_ptr_local = int(local_tensor.data_ptr())
                    if self.copy_engine_enabled:
                        launch_p2p_copy_engine(
                            src_ptr=src_ptr_remote,
                            dst_ptr=dst_ptr_local,
                            nbytes=nbytes,
                            stream=self.stream,
                        )
                    elif self.tma_enabled and nbytes % 16 == 0 and dst_ptr_local % 16 == 0:
                        launch_p2p_copy_tma(
                            src_ptr=src_ptr_remote,
                            dst_ptr=dst_ptr_local,
                            nbytes=nbytes,
                            num_sms=self.num_sms,
                        )
                    else:
                        vec = _vec_for(src_ptr_remote, dst_ptr_local, nbytes)
                        launch_pull_copy(
                            src_ptr=src_ptr_remote,
                            dst_ptr=dst_ptr_local,
                            nbytes=nbytes,
                            vec_bytes=vec,
                            num_sms=self.copy_sms,
                        )
                # else: this rank participates in barriers only.

                self._barrier_on(group, row_ptr, post_tok)

                # Receiver in push protocol pulls from own comm_buf into dst tensor.
                if protocol == "put" and role == "receiver":
                    local_buf = self.regions.comm_buf.tensor[
                        comm_buf_offset : comm_buf_offset + nbytes
                    ]
                    local_tensor.view(torch.uint8).reshape(-1)[:nbytes].copy_(local_buf)

    # -------------------------------------------------- autotune dispatch
    def _autotune_path_guard(
        self,
        policy: Policy,
        slice_bytes: int,
        *,
        comm_buf_offset: int,
        dst_ptr: int,
    ) -> None:
        """Reject autotune policies the runtime is not provisioned for.

        A JSON config or a programmatically-installed table can name a path
        whose preconditions the runtime never established. Catch those at the
        dispatch boundary, **before any epoch is consumed or kernel launched**,
        so a rejected policy leaves the runtime's per-group epoch counter and
        signal slots untouched (a half-issued collective — pre-barrier enqueued,
        epochs spent — would otherwise perturb the per-edge slot sequence):

        * ``tma`` requires the init-time peer-pointer probe to have passed
          (``self.tma_enabled``) and the specific call's alignment gates to
          pass; otherwise the TMA launcher would raise after the pre-barrier.
        * ``fused`` knobs are validated in full here. ``step_pad`` is sized once
          from ``config.fused_num_channels`` × ``fused_max_chunks_per_channel``
          (and rendezvoused at that size on every rank), so the policy's channel
          count must fit it and the resulting chunk count must fit the static
          chunk bound — resizing per-policy is unsafe (the symmetric region must
          be identically sized across ranks). ``fused_chunk_size`` must also be
          alignment-multiple: the kernel places chunk ``k`` at byte
          ``k * fused_chunk_size`` and stores it vectorized, so a non-aligned
          chunk size misaligns every k>0 store. The launcher re-checks the chunk
          count as a defence-in-depth backstop, but only *after* the pre-barrier
          is already enqueued — hence the full preflight here.
        """
        if policy.path == "tma":
            _require(
                self.tma_enabled,
                "autotune selected path 'tma' but TMA is not enabled "
                "(use_tma=False or the TMA probe did not pass); refusing to "
                "launch the TMA kernel on an unverified path",
            )
            self._preflight_tma_collective(
                slice_bytes=slice_bytes,
                comm_buf_offset=comm_buf_offset,
                dst_ptr=dst_ptr,
            )
        elif policy.path == "fused":
            cfg = self.config
            num_channels = int(
                policy.knobs.get("fused_num_channels", cfg.fused_num_channels)
            )
            chunk_size = int(
                policy.knobs.get("fused_chunk_size", cfg.fused_chunk_size)
            )
            _require(num_channels > 0, "autotune fused_num_channels must be > 0")
            self._check_fused_launch_capacity(num_channels)
            _require(
                num_channels <= cfg.fused_num_channels,
                f"autotune fused_num_channels={num_channels} exceeds the "
                f"allocated config.fused_num_channels={cfg.fused_num_channels}; "
                "step_pad is sized from config.fused_num_channels (it backs the "
                "symmetric step grid) — raise it in the runtime config to use "
                "more channels",
            )
            _require(chunk_size > 0, "autotune fused_chunk_size must be > 0")
            _require(
                chunk_size % cfg.alignment == 0,
                f"autotune fused_chunk_size={chunk_size} must be a multiple of "
                f"alignment={cfg.alignment}; otherwise chunk bases at "
                f"k*fused_chunk_size lose vector alignment and the fused "
                f"kernel's vectorized stores fault",
            )
            # Reproduce the launcher's per-channel chunk count so we reject here
            # (before epochs/pre-barrier) rather than at the launcher's backstop.
            partition_bytes = (slice_bytes + num_channels - 1) // num_channels
            partition_bytes = (
                (partition_bytes + cfg.alignment - 1) & ~(cfg.alignment - 1)
            )
            n_chunks = (partition_bytes + chunk_size - 1) // chunk_size
            _require(
                n_chunks <= cfg.fused_max_chunks_per_channel,
                f"autotune fused config yields {n_chunks} chunks/channel for "
                f"slice_bytes={slice_bytes} (channels={num_channels}, "
                f"chunk_size={chunk_size}) > static bound "
                f"{cfg.fused_max_chunks_per_channel}; raise fused_chunk_size or "
                f"max_collective_bytes",
            )

    def _dispatch_all_gather(
        self,
        *,
        policy: Policy,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
    ) -> None:
        """Route an all_gather call according to an autotune policy.

        Each branch issues its own pre/post barriers; nothing is returned.
        The fused kernel itself handles any alignment (picks a vec width
        of 16/8/4/1 plus a byte-tail at runtime), so no eligibility check
        is needed here. ``pipelined`` falls back to ``vec_pull`` when
        the chunk count collapses to 1.
        """
        self._autotune_path_guard(
            policy,
            slice_bytes,
            comm_buf_offset=comm_buf_offset,
            dst_ptr=int(output.data_ptr()),
        )
        path = policy.path
        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            if chunks <= 1:
                path = "vec_pull"

        if path == "fused":
            num_channels = int(
                policy.knobs.get("fused_num_channels", self.config.fused_num_channels)
            )
            chunk_size = int(
                policy.knobs.get("fused_chunk_size", self.config.fused_chunk_size)
            )
            pre_e = self._epochs.next_for_barrier(group.group_id)
            step_base_e = self._epochs.next_for_barrier(group.group_id)
            post_e = self._epochs.next_for_barrier(group.group_id)
            pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
            step_base = compute_token(
                self.session_nonce, group.group_id, step_base_e
            )
            post_tok = compute_token(self.session_nonce, group.group_id, post_e)
            with torch.cuda.stream(self.stream):
                self._barrier_on(group, row_ptr, pre_tok)
                launch_all_gather_fused(
                    input_ptr=int(input_.data_ptr()),
                    output_ptr=int(output.data_ptr()),
                    comm_buf_ptrs_dev=cb_ptrs,
                    step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_global_rank=self.rank,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    world_size=self.world_size,
                    max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                    num_channels=num_channels,
                    chunk_bytes=chunk_size,
                    step_base=step_base,
                    alignment=self.config.alignment,
                    timeout_ns=self._timeout_ns,
                )
                # Post-barrier: prove every peer finished pulling this rank's
                # comm_buf before a later collective reuses it (the fused step
                # token only proves *this* rank's own pulls completed).
                self._barrier_on(group, row_ptr, post_tok)
            return None

        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            self._all_gather_pipelined(
                input_=input_,
                output=output,
                group=group,
                comm_buf_offset=comm_buf_offset,
                row_ptr=row_ptr,
                sig_ptrs=sig_ptrs,
                cb_ptrs=cb_ptrs,
                slice_bytes=slice_bytes,
                chunks=chunks,
            )
            return None

        # One-shot stage + barrier + body + barrier paths.
        pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
        pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)
        with torch.cuda.stream(self.stream):
            local_buf = self.regions.comm_buf.tensor[
                comm_buf_offset : comm_buf_offset + slice_bytes
            ]
            local_buf.copy_(input_.view(torch.uint8).reshape(-1))
            self._barrier_on(group, row_ptr, pre_tok)
            if path == "copy_engine":
                launch_all_gather_copy_engine(
                    peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks=group.ranks,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    stream=self.stream,
                )
            elif path == "tma":
                launch_all_gather_tma(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    num_sms=self.num_sms,
                )
            else:
                copy_sms = int(policy.knobs.get("copy_sms", self.copy_sms))
                vec = _vec_for(
                    self.regions.comm_buf.local_ptr,
                    int(output.data_ptr()),
                    slice_bytes,
                )
                launch_all_gather_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=copy_sms,
                )
            self._barrier_on(group, row_ptr, post_tok)
        return None

    def _dispatch_all2all(
        self,
        *,
        policy: Policy,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
        staged_bytes: int,
    ) -> None:
        self._autotune_path_guard(
            policy,
            slice_bytes,
            comm_buf_offset=comm_buf_offset,
            dst_ptr=int(output.data_ptr()),
        )
        path = policy.path
        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            if chunks <= 1:
                path = "vec_pull"

        if path == "fused":
            num_channels = int(
                policy.knobs.get("fused_num_channels", self.config.fused_num_channels)
            )
            chunk_size = int(
                policy.knobs.get("fused_chunk_size", self.config.fused_chunk_size)
            )
            pre_e = self._epochs.next_for_barrier(group.group_id)
            step_base_e = self._epochs.next_for_barrier(group.group_id)
            post_e = self._epochs.next_for_barrier(group.group_id)
            pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
            step_base = compute_token(
                self.session_nonce, group.group_id, step_base_e
            )
            post_tok = compute_token(self.session_nonce, group.group_id, post_e)
            with torch.cuda.stream(self.stream):
                self._barrier_on(group, row_ptr, pre_tok)
                launch_all2all_fused(
                    input_ptr=int(input_.data_ptr()),
                    output_ptr=int(output.data_ptr()),
                    comm_buf_ptrs_dev=cb_ptrs,
                    step_pad_ptrs_dev=self.step_pad_ptrs_dev,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_global_rank=self.rank,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    world_size=self.world_size,
                    max_chunks_per_channel=self.config.fused_max_chunks_per_channel,
                    num_channels=num_channels,
                    chunk_bytes=chunk_size,
                    step_base=step_base,
                    alignment=self.config.alignment,
                    timeout_ns=self._timeout_ns,
                )
                # Post-barrier: prove every peer finished pulling this rank's
                # comm_buf before a later collective reuses it (the fused step
                # token only proves *this* rank's own pulls completed).
                self._barrier_on(group, row_ptr, post_tok)
            return None

        if path == "pipelined":
            chunks = self._effective_pipeline_chunks(
                slice_bytes, int(policy.knobs.get("pipeline_chunks", 2))
            )
            self._all2all_pipelined(
                input_=input_,
                output=output,
                group=group,
                comm_buf_offset=comm_buf_offset,
                row_ptr=row_ptr,
                sig_ptrs=sig_ptrs,
                cb_ptrs=cb_ptrs,
                slice_bytes=slice_bytes,
                chunks=chunks,
            )
            return None

        pre_e, post_e = self._epochs.next_pair_for_collective(group.group_id)
        pre_tok = compute_token(self.session_nonce, group.group_id, pre_e)
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)
        with torch.cuda.stream(self.stream):
            local_buf = self.regions.comm_buf.tensor[
                comm_buf_offset : comm_buf_offset + staged_bytes
            ]
            local_buf.copy_(
                input_.view(torch.uint8).reshape(-1)[:staged_bytes]
            )
            self._barrier_on(group, row_ptr, pre_tok)
            if path == "copy_engine":
                launch_all2all_copy_engine(
                    peer_comm_ptrs=self.regions.comm_buf.handle.buffer_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks=group.ranks,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    stream=self.stream,
                )
            elif path == "tma":
                launch_all2all_tma(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    num_sms=self.num_sms,
                )
            else:
                copy_sms = int(policy.knobs.get("copy_sms", self.copy_sms))
                vec = _vec_for(
                    self.regions.comm_buf.local_ptr,
                    int(output.data_ptr()),
                    slice_bytes,
                )
                launch_all2all_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=int(output.data_ptr()),
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=copy_sms,
                )
            self._barrier_on(group, row_ptr, post_tok)
        return None

    # ------------------------------------------------- pipelined helpers
    def _effective_pipeline_chunks(
        self, axis_bytes: int, override: Optional[int]
    ) -> int:
        """Resolve the pipeline chunk count for a collective.

        ``axis_bytes`` is the per-peer chunkable axis: ``input.nbytes`` for
        all_gather and ``slice_bytes`` (= input.nbytes / group.size) for
        all2all. Returns 1 when pipelining is disabled, when the data path
        does not yet support chunking (TMA / copy-engine), or when
        ``axis_bytes`` is too small for the configured minimum chunk size.
        """
        if self.tma_enabled or self.copy_engine_enabled:
            return 1
        target = override if override is not None else self.config.pipeline_chunks
        target = min(int(target), self.config.max_pipeline_chunks)
        if target <= 1:
            return 1
        min_chunk = self.config.pipeline_min_chunk_bytes
        if axis_bytes < 2 * min_chunk:
            return 1
        # Cap chunks so each chunk has at least ``min_chunk`` bytes.
        max_by_size = max(1, axis_bytes // min_chunk)
        return max(1, min(target, max_by_size))

    @staticmethod
    def _chunk_layout(axis_bytes: int, chunks: int, align: int = 16) -> list[tuple[int, int]]:
        """Return ``[(offset, size)]`` covering ``axis_bytes`` in up to
        ``chunks`` aligned slices.

        The first ``chunks - 1`` chunks are aligned up to ``align`` bytes;
        the last chunk absorbs the remainder. This keeps every per-peer
        sub-buffer 16B-aligned so the vec kernel keeps its widest store.
        """
        _require(chunks >= 1, "chunks must be >= 1")
        if chunks == 1 or axis_bytes <= 0:
            return [(0, axis_bytes)]
        nominal = (axis_bytes + chunks - 1) // chunks
        nominal = (nominal + align - 1) & ~(align - 1)
        layout: list[tuple[int, int]] = []
        off = 0
        while off < axis_bytes and len(layout) < chunks:
            size = min(nominal, axis_bytes - off)
            layout.append((off, size))
            off += size
        # If alignment rounding caused us to skip some bytes, absorb the
        # remainder into the last chunk.
        if off < axis_bytes:
            last_off, last_size = layout[-1]
            layout[-1] = (last_off, last_size + (axis_bytes - off))
        return layout

    def _all_gather_pipelined(
        self,
        *,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
        chunks: int,
    ) -> None:
        """Chunked all_gather; issues its own pre-barriers and post-barrier."""
        layout = self._chunk_layout(slice_bytes, chunks, align=self.config.alignment)
        actual_chunks = len(layout)
        max_chunks = self.config.max_pipeline_chunks
        _require(
            actual_chunks <= max_chunks,
            f"chunked all_gather: actual_chunks={actual_chunks} > max={max_chunks}",
        )

        # Reserve one epoch per pre-barrier chunk + one for the post.
        pre_epochs = [
            self._epochs.next_for_barrier(group.group_id) for _ in range(actual_chunks)
        ]
        post_e = self._epochs.next_for_barrier(group.group_id)
        pre_toks = [
            compute_token(self.session_nonce, group.group_id, e) for e in pre_epochs
        ]
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)

        # Fork copy_stream off main_stream. The prior pulls and post-barrier are
        # already in main_stream's history, so waiting on this event blocks
        # staging until that prior work is done.
        fork_ev = self.stream.record_event()
        self.copy_stream.wait_event(fork_ev)

        src_u8 = input_.view(torch.uint8).reshape(-1)
        cb_local = self.regions.comm_buf.tensor
        output_ptr = int(output.data_ptr())
        dst_stride = slice_bytes  # peer-stride in output is full slice_bytes
        comm_buf_local_ptr = self.regions.comm_buf.local_ptr

        for k, (offset, size) in enumerate(layout):
            # Stage chunk k on copy_stream.
            with torch.cuda.stream(self.copy_stream):
                dst_buf = cb_local[
                    comm_buf_offset + offset : comm_buf_offset + offset + size
                ]
                dst_buf.copy_(src_u8[offset : offset + size])
            stage_ev = self.copy_stream.record_event()

            with torch.cuda.stream(self.stream):
                self.stream.wait_event(stage_ev)
                self._barrier_on(group, row_ptr, pre_toks[k])

                vec = _vec_for(
                    comm_buf_local_ptr + offset,
                    output_ptr + offset,
                    # Peers land at ``output + peer_idx * dst_stride``; the vec
                    # width must divide that stride too, not just the chunk
                    # base, or peer_idx>=1 stores misalign when slice_bytes is
                    # not 16/8/4-aligned.
                    dst_stride,
                    size,
                )
                launch_all_gather_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=output_ptr + offset,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset + offset,
                    self_local_index=group.local_index,
                    slice_bytes=size,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=self.copy_sms,
                    dst_stride_bytes=dst_stride,
                )

        with torch.cuda.stream(self.stream):
            self._barrier_on(group, row_ptr, post_tok)

        return None

    def _all2all_pipelined(
        self,
        *,
        input_: torch.Tensor,
        output: torch.Tensor,
        group: GroupDescriptor,
        comm_buf_offset: int,
        row_ptr: int,
        sig_ptrs: int,
        cb_ptrs: int,
        slice_bytes: int,
        chunks: int,
    ) -> None:
        """Chunked all2all; issues its own pre-barriers and post-barrier.

        For all2all every chunk is a sub-slice of size ``chunk_bytes`` within
        each per-peer slice. The local stage of chunk k copies a single
        contiguous run of ``group.size * chunk_bytes`` bytes by issuing one
        copy per peer slice to the comm_buf at offset
        ``comm_buf_offset + peer_idx * slice_bytes + chunk_offset``.
        """
        layout = self._chunk_layout(slice_bytes, chunks, align=self.config.alignment)
        actual_chunks = len(layout)
        max_chunks = self.config.max_pipeline_chunks
        _require(
            actual_chunks <= max_chunks,
            f"chunked all2all: actual_chunks={actual_chunks} > max={max_chunks}",
        )

        pre_epochs = [
            self._epochs.next_for_barrier(group.group_id) for _ in range(actual_chunks)
        ]
        post_e = self._epochs.next_for_barrier(group.group_id)
        pre_toks = [
            compute_token(self.session_nonce, group.group_id, e) for e in pre_epochs
        ]
        post_tok = compute_token(self.session_nonce, group.group_id, post_e)

        fork_ev = self.stream.record_event()
        self.copy_stream.wait_event(fork_ev)

        src_u8 = input_.view(torch.uint8).reshape(-1)
        cb_local = self.regions.comm_buf.tensor
        output_ptr = int(output.data_ptr())
        comm_buf_local_ptr = self.regions.comm_buf.local_ptr
        group_size = group.size

        for k, (offset, size) in enumerate(layout):
            # Stage chunk k on copy_stream: copy the chunk-bytes destined for
            # each peer from input[peer*slice_bytes + offset : ...] into
            # comm_buf[peer*slice_bytes + offset : ...]. The runs
            # within input and within comm_buf are not contiguous across
            # peers, so we issue ``group_size`` copies per chunk.
            with torch.cuda.stream(self.copy_stream):
                for peer_idx in range(group_size):
                    in_off = peer_idx * slice_bytes + offset
                    cb_off = comm_buf_offset + peer_idx * slice_bytes + offset
                    cb_local[cb_off : cb_off + size].copy_(
                        src_u8[in_off : in_off + size]
                    )
            stage_ev = self.copy_stream.record_event()

            with torch.cuda.stream(self.stream):
                self.stream.wait_event(stage_ev)
                self._barrier_on(group, row_ptr, pre_toks[k])

                vec = _vec_for(
                    comm_buf_local_ptr
                    + comm_buf_offset
                    + group.local_index * slice_bytes
                    + offset,
                    output_ptr + offset,
                    # src/dst both stride by per-peer ``slice_bytes`` inside the
                    # kernel; the vec width must divide that stride so peer
                    # segments stay aligned when slice_bytes is not 16-aligned.
                    slice_bytes,
                    size,
                )
                launch_all2all_pull(
                    comm_buf_ptrs_dev=cb_ptrs,
                    dst_ptr=output_ptr + offset,
                    group_ranks_row_ptr=row_ptr,
                    comm_buf_offset=comm_buf_offset + offset,
                    self_local_index=group.local_index,
                    slice_bytes=slice_bytes,
                    group_size=group.size,
                    vec_bytes=vec,
                    num_sms=self.copy_sms,
                    copy_bytes=size,
                    src_slice_stride=slice_bytes,
                    dst_stride_bytes=slice_bytes,
                )

        with torch.cuda.stream(self.stream):
            self._barrier_on(group, row_ptr, post_tok)

        return None

    # --------------------------------------------- external stream integration
    def wait_for_external(self, event: torch.cuda.Event) -> None:
        """Insert a one-way edge: ``self.stream`` waits on ``event``."""
        self.stream.wait_event(event)

    def record_event(self, event: Optional[torch.cuda.Event] = None) -> torch.cuda.Event:
        """Record a completion event on ``self.stream`` and return it."""
        ev = event if event is not None else torch.cuda.Event()
        ev.record(self.stream)
        return ev

    # -------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        """Drain, zero the signal/step grids, free group rank lists, world-barrier.

        After ``shutdown`` the runtime object is no longer usable for
        collectives. Repeated shutdown is a no-op.
        """
        if getattr(self, "_shutdown", False):
            return
        torch.cuda.current_stream(self.device).synchronize()
        self.stream.synchronize()
        self.copy_stream.synchronize()
        zero_signal_grids(self.regions)
        self.edge_seq.zero_()
        # Streams are drained, so no in-flight kernel still references the
        # per-group rank lists; drop them directly (no record_stream needed) so
        # a long-lived process that registered many groups without explicitly
        # unregistering them reclaims that device memory at shutdown.
        for desc in self._registered.values():
            if desc.ranks_dev is not None:
                object.__setattr__(desc, "ranks_dev", None)
        self._registered.clear()
        dist.barrier(group=self.world_pg)
        self._shutdown = True

    # -------------------------------------------------------------- accessors
    @property
    def comm_buf_ptrs_dev(self) -> int:
        return self.regions.comm_buf.buffer_ptrs_dev

    @property
    def signal_buf_ptrs_dev(self) -> int:
        return self.regions.signal_buf.buffer_ptrs_dev

    @property
    def edge_seq_ptr(self) -> int:
        return int(self.edge_seq.data_ptr())

    @property
    def step_pad_ptrs_dev(self) -> int:
        return self.regions.step_pad.buffer_ptrs_dev


# --------------------------------------------------------------------- helpers
def _validate_collective_args(
    input_: torch.Tensor,
    output: torch.Tensor,
    group: GroupDescriptor,
    device: torch.device,
    config: SymmetricCollectiveConfig,
    *,
    kind: str,
) -> None:
    _require(group.local_index >= 0, f"this rank is not a member of group {group.ranks}")
    _require(
        group.size <= config.max_group_size,
        f"group.size={group.size} > max_group_size={config.max_group_size}",
    )
    _require(input_.is_cuda and output.is_cuda, f"{kind}: tensors must be on CUDA")
    _require(input_.device == device, f"{kind}: input on {input_.device}, runtime on {device}")
    _require(output.device == device, f"{kind}: output on {output.device}, runtime on {device}")
    _require(input_.is_contiguous(), f"{kind}: input must be contiguous")
    _require(output.is_contiguous(), f"{kind}: output must be contiguous")
    if kind == "all_gather":
        _require(
            input_.nbytes <= config.max_collective_bytes,
            f"{kind}: input.nbytes {input_.nbytes} > max_collective_bytes "
            f"{config.max_collective_bytes}",
        )
    elif kind == "all2all":
        _require(
            output.nbytes <= config.max_collective_bytes,
            f"{kind}: output.nbytes {output.nbytes} > max_collective_bytes "
            f"{config.max_collective_bytes}",
        )


def _vec_for(*ptrs_and_size: int) -> int:
    """Pick a vector width given pointer addresses and a payload size."""
    aligns = []
    for v in ptrs_and_size[:-1]:
        aligns.append(v & -v if v else 1 << 62)
    aligns.append(int(ptrs_and_size[-1]))
    return vec_width_bytes(*aligns)


def _replace(c: SymmetricCollectiveConfig, **fields) -> SymmetricCollectiveConfig:
    """``dataclasses.replace`` but without re-importing it everywhere."""
    import dataclasses

    return dataclasses.replace(c, **fields)


# ---------------------------------------------------------------- conveniences
def init_distributed_for_runtime() -> torch.device:
    """Initialize the bootstrap NCCL process group from torchrun env vars.

    Returns the CUDA device for this rank (``cuda:LOCAL_RANK``).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return device
