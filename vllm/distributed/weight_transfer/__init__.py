# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Weight transfer engines for syncing model weights between processes.

Includes:
- Train-to-inference weight transfer (WeightTransferEngine, factory)
- Live P2P weight transfer between vLLM instances (packed_buffer, server)
"""

from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.distributed.weight_transfer.packed_buffer import (
    DEFAULT_BUFFER_SIZE,
    PackedMetadata,
    TensorMeta,
    compute_buffer_plan,
)

__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "PackedMetadata",
    "TensorMeta",
    "WeightTransferEngineFactory",
    "compute_buffer_plan",
]
