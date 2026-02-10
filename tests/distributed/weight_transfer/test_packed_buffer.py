# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for deterministic buffer packing algorithm."""

import pytest

from vllm.distributed.weight_transfer.packed_buffer import (
    DEFAULT_BUFFER_SIZE,
    PackedMetadata,
    TensorMeta,
    compute_buffer_plan,
)


class TestComputeBufferPlan:
    """Tests for compute_buffer_plan()."""

    def test_empty_input(self):
        """Empty tensor list produces no buffers."""
        plan = compute_buffer_plan([], buffer_size=1024)
        assert plan == []

    def test_single_tensor_fits(self):
        """Single tensor smaller than buffer_size goes into one buffer."""
        metas = [("w", [4, 4], "torch.float32", 64)]
        plan = compute_buffer_plan(metas, buffer_size=128)
        assert len(plan) == 1
        buf_bytes, assignments = plan[0]
        assert buf_bytes == 64
        assert assignments == [(0, 0)]

    def test_two_tensors_fit_one_buffer(self):
        """Two small tensors pack into a single buffer."""
        metas = [
            ("w1", [4, 4], "torch.float32", 64),
            ("w2", [8], "torch.float32", 32),
        ]
        plan = compute_buffer_plan(metas, buffer_size=128)
        assert len(plan) == 1
        buf_bytes, assignments = plan[0]
        assert buf_bytes == 96
        assert assignments == [(0, 0), (1, 64)]

    def test_tensors_split_across_buffers(self):
        """Tensors that exceed buffer_size are split into multiple buffers."""
        metas = [
            ("w1", [4, 4], "torch.float32", 64),
            ("w2", [4, 4], "torch.float32", 64),
            ("w3", [4, 4], "torch.float32", 64),
        ]
        plan = compute_buffer_plan(metas, buffer_size=100)
        # w1 fits in buf0, w2 doesn't fit (64+64>100), starts buf1,
        # w3 doesn't fit (64+64>100), starts buf2
        assert len(plan) == 3
        assert plan[0] == (64, [(0, 0)])
        assert plan[1] == (64, [(1, 0)])
        assert plan[2] == (64, [(2, 0)])

    def test_oversized_tensor_gets_dedicated_buffer(self):
        """A tensor larger than buffer_size gets its own buffer."""
        metas = [
            ("small", [4], "torch.float32", 16),
            ("huge", [1024, 1024], "torch.float32", 4_194_304),
            ("small2", [4], "torch.float32", 16),
        ]
        plan = compute_buffer_plan(metas, buffer_size=1024)
        assert len(plan) == 3
        # First buffer: small tensor
        assert plan[0] == (16, [(0, 0)])
        # Second buffer: oversized tensor alone
        assert plan[1] == (4_194_304, [(1, 0)])
        # Third buffer: small2
        assert plan[2] == (16, [(2, 0)])

    def test_exact_fit(self):
        """Tensors that exactly fill a buffer stay in one buffer."""
        metas = [
            ("w1", [32], "torch.float32", 128),
            ("w2", [32], "torch.float32", 128),
        ]
        plan = compute_buffer_plan(metas, buffer_size=256)
        assert len(plan) == 1
        assert plan[0] == (256, [(0, 0), (1, 128)])

    def test_exact_overflow(self):
        """Adding one byte over buffer_size triggers a new buffer."""
        metas = [
            ("w1", [32], "torch.float32", 128),
            ("w2", [33], "torch.float32", 129),
        ]
        # 128 + 129 = 257 > 256, so w2 goes to new buffer
        plan = compute_buffer_plan(metas, buffer_size=256)
        assert len(plan) == 2
        assert plan[0] == (128, [(0, 0)])
        assert plan[1] == (129, [(1, 0)])

    def test_deterministic(self):
        """Same inputs always produce identical plans."""
        metas = [
            ("w1", [100], "torch.float32", 400),
            ("w2", [200], "torch.bfloat16", 400),
            ("w3", [50], "torch.int8", 50),
        ]
        plan1 = compute_buffer_plan(metas, buffer_size=500)
        plan2 = compute_buffer_plan(metas, buffer_size=500)
        assert plan1 == plan2

    def test_offsets_are_contiguous(self):
        """Within a buffer, offsets are contiguous (no gaps)."""
        metas = [
            ("w1", [10], "torch.float32", 40),
            ("w2", [20], "torch.float32", 80),
            ("w3", [5], "torch.float32", 20),
        ]
        plan = compute_buffer_plan(metas, buffer_size=1024)
        assert len(plan) == 1
        buf_bytes, assignments = plan[0]
        assert buf_bytes == 140
        assert assignments[0] == (0, 0)
        assert assignments[1] == (1, 40)
        assert assignments[2] == (2, 120)


class TestTensorMeta:
    """Tests for TensorMeta dataclass."""

    def test_to_tuple(self):
        """to_tuple produces wire-format tuple."""
        meta = TensorMeta(
            name="model.weight",
            shape=[4, 4],
            dtype_str="torch.float32",
            nbytes=64,
        )
        assert meta.to_tuple() == ("model.weight", [4, 4], "torch.float32", 64)

    def test_frozen(self):
        """TensorMeta is immutable."""
        meta = TensorMeta("w", [4], "torch.float32", 16)
        with pytest.raises(AttributeError):
            meta.name = "other"


class TestPackedMetadata:
    """Tests for PackedMetadata dataclass."""

    def test_roundtrip(self):
        """to_dict/from_dict roundtrip preserves all fields."""
        original = PackedMetadata(
            version=2,
            tensors=[("w1", [4, 4], "torch.float32", 64)],
            buffer_size=2 * 1024**3,
            format="state_dict",
        )
        d = original.to_dict()
        restored = PackedMetadata.from_dict(d)
        assert restored.version == original.version
        assert restored.tensors == original.tensors
        assert restored.buffer_size == original.buffer_size
        assert restored.format == original.format

    def test_default_format(self):
        """from_dict defaults format to 'state_dict' if missing."""
        d = {"version": 2, "tensors": [], "buffer_size": 1024}
        meta = PackedMetadata.from_dict(d)
        assert meta.format == "state_dict"


class TestDefaultBufferSize:
    """Tests for DEFAULT_BUFFER_SIZE constant."""

    def test_is_2gb(self):
        assert DEFAULT_BUFFER_SIZE == 2 * 1024**3
