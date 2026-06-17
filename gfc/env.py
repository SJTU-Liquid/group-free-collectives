"""Environment variable handling for the ``SYMM_COLL_*`` flags.

See Section 11 of the design spec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: Optional[int]) -> Optional[int]:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


@dataclass(frozen=True)
class _EnvOverrides:
    debug: bool
    use_tma: bool
    use_copy_engine: bool
    copy_sms: Optional[int]
    timeout_ms: Optional[int]
    dump_signals: bool
    pipeline_chunks: Optional[int]
    max_pipeline_chunks: Optional[int]
    enable_fused_path: Optional[bool]
    fused_num_channels: Optional[int]
    fused_chunk_size: Optional[int]
    autotune_config_path: Optional[str]


def read_env_overrides() -> _EnvOverrides:
    fused_env = os.environ.get("SYMM_COLL_ENABLE_FUSED")
    fused_val: Optional[bool] = None
    if fused_env is not None:
        fused_val = fused_env.strip().lower() in ("1", "true", "yes", "on")
    return _EnvOverrides(
        debug=_bool_env("SYMM_COLL_DEBUG"),
        use_tma=_bool_env("SYMM_COLL_USE_TMA"),
        use_copy_engine=_bool_env("SYMM_COLL_USE_COPY_ENGINE"),
        copy_sms=_int_env("SYMM_COLL_COPY_SMS", None),
        timeout_ms=_int_env("SYMM_COLL_TIMEOUT_MS", None),
        dump_signals=_bool_env("SYMM_COLL_DUMP_SIGNALS"),
        pipeline_chunks=_int_env("SYMM_COLL_PIPELINE_CHUNKS", None),
        max_pipeline_chunks=_int_env("SYMM_COLL_MAX_PIPELINE_CHUNKS", None),
        enable_fused_path=fused_val,
        fused_num_channels=_int_env("SYMM_COLL_FUSED_NUM_CHANNELS", None),
        fused_chunk_size=_int_env("SYMM_COLL_FUSED_CHUNK_SIZE", None),
        autotune_config_path=os.environ.get("SYMM_COLL_AUTOTUNE_CONFIG") or None,
    )
