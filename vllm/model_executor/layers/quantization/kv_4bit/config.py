from __future__ import annotations

import os
from dataclasses import dataclass


KV_4BIT_DTYPES = ("kv_4bit",)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as err:
        raise ValueError(f"{name} must be an integer, got {value!r}") from err


@dataclass(frozen=True)
class KV4BitConfig:
    head_dim: int
    hadamard_order: int = 128

    @property
    def data_bytes(self) -> int:
        return self.head_dim // 2

    @property
    def scale_zero_bytes(self) -> int:
        # float32 scale + float32 zero, stored inline as bytes.
        return 8

    @property
    def key_packed_size(self) -> int:
        return self.data_bytes + self.scale_zero_bytes

    @property
    def value_packed_size(self) -> int:
        return self.data_bytes + self.scale_zero_bytes

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        # Direct fp32 scale/zero cache aliases require 4-byte-aligned slots.
        return self.slot_size + (-self.slot_size % 4)

    def validate(self) -> None:
        if self.head_dim % 8:
            raise ValueError(
                f"KV-4BIT requires head_dim to be a multiple of 8, "
                f"got {self.head_dim}"
            )
        order = self.hadamard_order
        if order < 2:
            raise ValueError(f"Hadamard order must be >= 2, got {order}")
        if order & (order - 1):
            raise ValueError(f"Hadamard order must be a power of two, got {order}")
        if self.head_dim % order:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be divisible by "
                f"Hadamard order ({order})"
            )

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str, head_dim: int) -> "KV4BitConfig":
        if cache_dtype not in KV_4BIT_DTYPES:
            valid = ", ".join(KV_4BIT_DTYPES)
            raise ValueError(
                f"Unknown KV-4BIT cache dtype {cache_dtype!r}. Valid: {valid}"
            )
        cfg = cls(
            head_dim=head_dim,
            hadamard_order=_env_int("VLLM_KV_4BIT_HADAMARD_ORDER", 128),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_mla_cache_dtype(cls, cache_dtype: str, head_dim: int) -> "KV4BitConfig":
        if cache_dtype not in KV_4BIT_DTYPES:
            valid = ", ".join(KV_4BIT_DTYPES)
            raise ValueError(
                f"Unknown KV-4BIT cache dtype {cache_dtype!r}. Valid: {valid}"
            )
        cfg = cls(
            head_dim=head_dim,
            hadamard_order=_env_int("VLLM_KV_4BIT_MLA_HADAMARD_ORDER", 64),
        )
        cfg.validate()
        return cfg
