"""Autotune policy table for collective dispatch.

A *policy* maps ``(collective, group_size, slice_bytes)`` to a chosen
data-path + path-specific knobs. The table is a list of rules; the first
rule whose pattern matches wins. Rules should be sorted from most
specific to least specific (catch-all last).

Paths covered by this version:

* ``vec_pull`` — legacy pull kernel + separate barrier kernels (not-fused)
* ``fused``    — fused single-kernel pull (CTA-pair sender + receiver +
                 in-kernel step counter)
* ``tma``      — TMA-variant pull kernel (not-fused)
* ``copy_engine`` — host-enqueued ``cudaMemcpyAsync`` per peer
* ``pipelined``  — chunked stream-split pipeline (not-fused)

A policy may carry path-specific knobs:

* ``fused``: ``fused_num_channels``, ``fused_chunk_size``
* ``vec_pull``/``tma``: ``copy_sms``
* ``pipelined``: ``pipeline_chunks``

JSON schema (version 1):

.. code-block:: json

    {
      "version": 1,
      "platform": "NVIDIA H100 80GB HBM3",
      "world_size": 4,
      "rules": [
        {
          "collective": "all2all", "group_size": 4,
          "min_bytes": 67108864,
          "config": {"path": "fused", "fused_num_channels": 32, "fused_chunk_size": 262144}
        },
        {
          "collective": "*", "group_size": "*", "min_bytes": 0,
          "config": {"path": "vec_pull"}
        }
      ]
    }

Match semantics: a rule matches when ``collective`` (or ``"*"``) ==
queried collective, ``group_size`` (or ``"*"``) == queried group size,
and ``min_bytes <= slice_bytes <= max_bytes`` (``max_bytes`` defaults
to infinity). The first matching rule's ``config`` is returned.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


_KNOWN_PATHS = ("vec_pull", "fused", "tma", "copy_engine", "pipelined")


@dataclass(frozen=True)
class Policy:
    """A path selection + path-specific knob overrides for a single call."""

    path: str
    knobs: dict[str, Any]


class AutotuneTable:
    """In-memory representation of the autotune JSON.

    Use :meth:`from_json` to load a tunable from disk; use
    :meth:`policy_for` to dispatch a call.
    """

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self.rules = list(rules)
        for r in self.rules:
            cfg = r.get("config") or {}
            path = cfg.get("path")
            if path is not None and path not in _KNOWN_PATHS:
                raise ValueError(
                    f"autotune rule references unknown path {path!r}; "
                    f"expected one of {_KNOWN_PATHS}"
                )

    @classmethod
    def from_json(cls, path: str) -> "AutotuneTable":
        with open(path) as fh:
            obj = json.load(fh)
        if int(obj.get("version", 1)) != 1:
            raise ValueError(
                f"unsupported autotune config version {obj.get('version')}; "
                f"this build expects 1"
            )
        return cls(obj.get("rules", []))

    def max_fused_num_channels(self) -> int:
        """Largest ``fused_num_channels`` any fused rule requests (0 if none).

        The runtime sizes the symmetric ``step_pad`` grid once from
        ``config.fused_num_channels`` and cannot grow it per call, so a rule
        asking for more channels than the runtime is provisioned for is rejected
        at dispatch — but only once a payload large enough to select that rule
        actually arrives. Surfacing the max here lets the runtime validate the
        whole table up front and fail with an actionable message at load time
        rather than mid-run on a large bucket.
        """
        mx = 0
        for r in self.rules:
            cfg = r.get("config") or {}
            if cfg.get("path") == "fused":
                mx = max(mx, int(cfg.get("fused_num_channels", 0)))
        return mx

    def policy_for(
        self, collective: str, group_size: int, slice_bytes: int
    ) -> Policy | None:
        for rule in self.rules:
            r_coll = rule.get("collective", "*")
            r_gs = rule.get("group_size", "*")
            r_min = int(rule.get("min_bytes", 0))
            r_max_raw = rule.get("max_bytes", None)
            r_max = math.inf if r_max_raw is None else int(r_max_raw)
            if r_coll != "*" and r_coll != collective:
                continue
            if r_gs != "*" and int(r_gs) != int(group_size):
                continue
            if slice_bytes < r_min or slice_bytes > r_max:
                continue
            cfg = rule.get("config") or {}
            path = cfg.get("path")
            if not path:
                continue
            knobs = {k: v for k, v in cfg.items() if k != "path"}
            return Policy(path=path, knobs=knobs)
        return None
