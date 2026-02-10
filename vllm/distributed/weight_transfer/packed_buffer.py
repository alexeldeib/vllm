# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic buffer packing for NCCL weight transfer.

Provides a pre-computed buffer packing plan that both sender and receiver
can independently derive from the same tensor metadata. Unlike the streaming
approach in packed_tensor.py (which packs on-the-fly with double buffering),
this module computes the full layout ahead of time — enabling the sender to
pack from existing GPU tensors at known offsets without extra allocations.

Used by the weight server (donor) and NCCL weight loader (receiver) for
live P2P weight transfer between running vLLM instances.
"""

from dataclasses import dataclass

# 2 GB default — balances GPU memory pressure vs NCCL call overhead.
# At 47 GB/s IB line rate, 2 GB takes ~43ms per NCCL broadcast call.
DEFAULT_BUFFER_SIZE = 2 * 1024**3


@dataclass(frozen=True)
class TensorMeta:
    """Metadata for a single tensor in the transfer.

    Attributes:
        name: Parameter name (e.g., "model.layers.0.self_attn.q_proj.weight").
        shape: Tensor shape as a list of ints.
        dtype_str: String representation of dtype (e.g., "torch.bfloat16").
        nbytes: Total size in bytes.
    """

    name: str
    shape: list[int]
    dtype_str: str
    nbytes: int

    def to_tuple(self) -> tuple[str, list[int], str, int]:
        """Convert to wire-format tuple for JSON serialization."""
        return (self.name, self.shape, self.dtype_str, self.nbytes)


@dataclass(frozen=True)
class PackedMetadata:
    """Wire-format metadata sent from donor to receiver.

    Contains everything the receiver needs to independently compute
    the identical buffer packing plan.

    Attributes:
        version: Protocol version (currently 2).
        tensors: List of tensor metadata tuples
            ``(name, shape, dtype_str, nbytes)``.
        buffer_size: Buffer size in bytes used for packing.
        format: Describes the weight format ("state_dict" or "checkpoint").
    """

    version: int
    tensors: list[tuple[str, list[int], str, int]]
    buffer_size: int
    format: str = "state_dict"

    def to_dict(self) -> dict:
        """Convert to dict for broadcast_obj serialization."""
        return {
            "version": self.version,
            "tensors": self.tensors,
            "buffer_size": self.buffer_size,
            "format": self.format,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PackedMetadata":
        """Construct from received dict."""
        return cls(
            version=d["version"],
            tensors=d["tensors"],
            buffer_size=d["buffer_size"],
            format=d.get("format", "state_dict"),
        )


def compute_buffer_plan(
    tensor_metas: list[tuple[str, list[int], str, int]],
    buffer_size: int,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Compute a deterministic buffer packing plan.

    Both sender and receiver produce identical plans from the same inputs,
    so the packing layout never needs to be transmitted over the wire.

    Algorithm:
        - Tensors are packed sequentially into buffers of ``buffer_size``.
        - When adding a tensor would exceed the buffer, a new buffer starts.
        - Tensors larger than ``buffer_size`` get their own dedicated buffer.

    Args:
        tensor_metas: List of ``(name, shape, dtype_str, nbytes)`` tuples,
            one per tensor in transfer order.
        buffer_size: Maximum bytes per packed buffer.

    Returns:
        List of ``(buf_bytes, assignments)`` where ``assignments`` is a list
        of ``(tensor_index, byte_offset)`` pairs describing each tensor's
        position within the buffer.

    Example:
        >>> metas = [("w1", [4, 4], "torch.float32", 64),
        ...          ("w2", [8], "torch.float32", 32)]
        >>> plan = compute_buffer_plan(metas, buffer_size=128)
        >>> # Both tensors fit in one buffer:
        >>> assert len(plan) == 1
        >>> buf_bytes, assignments = plan[0]
        >>> assert buf_bytes == 96
        >>> assert assignments == [(0, 0), (1, 64)]
    """
    buffers: list[tuple[int, list[tuple[int, int]]]] = []
    current: list[tuple[int, int]] = []
    current_bytes = 0

    for i, (_name, _shape, _dtype, nbytes) in enumerate(tensor_metas):
        if nbytes > buffer_size:
            # Oversized tensor: flush current buffer, give it a dedicated one
            if current:
                buffers.append((current_bytes, current))
                current = []
                current_bytes = 0
            buffers.append((nbytes, [(i, 0)]))
        elif current_bytes + nbytes > buffer_size:
            # Current buffer is full, start a new one
            buffers.append((current_bytes, current))
            current = [(i, 0)]
            current_bytes = nbytes
        else:
            current.append((i, current_bytes))
            current_bytes += nbytes

    if current:
        buffers.append((current_bytes, current))

    return buffers
