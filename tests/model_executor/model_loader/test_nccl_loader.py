# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NCCL weight loader."""


import torch
import torch.nn as nn


class TestNCCLWeightLoaderConfig:
    """Test configuration and discovery logic (no GPU required)."""

    def test_register_format(self):
        """'nccl' format is registered in the loader registry."""
        from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER
        assert "nccl" in _LOAD_FORMAT_TO_MODEL_LOADER

    def test_config_defaults(self):
        """Default configuration values are sensible."""
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        loader = NCCLWeightLoader(LoadConfig(
            load_format="nccl",
            model_loader_extra_config={},
        ))
        assert loader._port == 29520
        assert loader._gpus_per_node == 8
        assert loader._mode == "sharded"
        assert loader._buffer_size == 2 * 1024**3
        assert loader._timeout == 300

    def test_config_override(self):
        """Extra config overrides defaults."""
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        loader = NCCLWeightLoader(LoadConfig(
            load_format="nccl",
            model_loader_extra_config={
                "nccl_addr": "10.0.0.1",
                "nccl_port": "12345",
                "gpus_per_node": "4",
                "buffer_size": "1073741824",
                "timeout": "60",
                "mode": "sharded",
            },
        ))
        assert loader._addr == "10.0.0.1"
        assert loader._port == 12345
        assert loader._gpus_per_node == 4
        assert loader._buffer_size == 1024**3
        assert loader._timeout == 60

    def test_multi_addr_parsing(self):
        """Comma-separated nccl_addrs are parsed correctly."""
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        loader = NCCLWeightLoader(LoadConfig(
            load_format="nccl",
            model_loader_extra_config={
                "nccl_addrs": "10.0.0.1, 10.0.0.2, 10.0.0.3",
            },
        ))
        assert loader._addrs == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    def test_empty_addrs(self):
        """Empty nccl_addrs string produces empty list."""
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        loader = NCCLWeightLoader(LoadConfig(
            load_format="nccl",
            model_loader_extra_config={"nccl_addrs": ""},
        ))
        assert loader._addrs == []


class TestAssignByPath:
    """Test _assign_by_path helper (no GPU required)."""

    def test_assign_direct(self):
        """Assign to a top-level parameter."""
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        model = nn.Linear(4, 4)
        new_data = torch.randn_like(model.weight.data)
        result = NCCLWeightLoader._assign_by_path(model, "weight", new_data)
        assert result is True
        assert torch.equal(model.weight.data, new_data)

    def test_assign_nested(self):
        """Assign to a nested parameter."""
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        model = nn.Sequential(nn.Linear(4, 4))
        new_data = torch.randn_like(model[0].weight.data)
        result = NCCLWeightLoader._assign_by_path(model, "0.weight", new_data)
        assert result is True
        assert torch.equal(model[0].weight.data, new_data)

    def test_assign_missing(self):
        """Missing path returns False."""
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        model = nn.Linear(4, 4)
        result = NCCLWeightLoader._assign_by_path(
            model, "nonexistent", torch.zeros(4)
        )
        assert result is False


class TestResolveDtype:
    """Test _resolve_dtype helper."""

    def test_common_dtypes(self):
        from vllm.model_executor.model_loader.nccl_loader import _resolve_dtype

        assert _resolve_dtype("torch.float16") == torch.float16
        assert _resolve_dtype("torch.bfloat16") == torch.bfloat16
        assert _resolve_dtype("torch.float32") == torch.float32
        assert _resolve_dtype("torch.int8") == torch.int8
        assert _resolve_dtype("torch.uint8") == torch.uint8

    def test_unknown_falls_back_to_uint8(self):
        from vllm.model_executor.model_loader.nccl_loader import _resolve_dtype

        assert _resolve_dtype("torch.nonexistent_dtype") == torch.uint8


class TestDiscoverDonor:
    """Test discovery routing logic."""

    def test_multi_addr_routing(self):
        """Multi-addr routes tp_rank to correct pod."""
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        loader = NCCLWeightLoader(LoadConfig(
            load_format="nccl",
            model_loader_extra_config={
                "nccl_addrs": "10.0.0.1,10.0.0.2",
                "gpus_per_node": "8",
            },
        ))

        # Mock _probe to return info for expected addr/port
        def mock_probe(addr, port):
            return {"nccl_port": port + 100, "status": "ready"}

        loader._probe = mock_probe

        # tp_rank 0 → pod 0 (10.0.0.1), local_rank 0, control port 29520
        result = loader._discover_donor(0)
        assert result["addr"] == "10.0.0.1"
        assert result["nccl_port"] == 29620

        # tp_rank 8 → pod 1 (10.0.0.2), local_rank 0, control port 29520
        result = loader._discover_donor(8)
        assert result["addr"] == "10.0.0.2"
        assert result["nccl_port"] == 29620

        # tp_rank 15 → pod 1 (10.0.0.2), local_rank 7, control port 29527
        result = loader._discover_donor(15)
        assert result["addr"] == "10.0.0.2"
        assert result["nccl_port"] == 29627

    def test_single_addr_fallback(self):
        """Falls back to single addr when multi-addr not configured."""
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        loader = NCCLWeightLoader(LoadConfig(
            load_format="nccl",
            model_loader_extra_config={"nccl_addr": "10.0.0.1"},
        ))
        loader._probe = lambda addr, port: {
            "nccl_port": port + 100, "status": "ready"
        }

        result = loader._discover_donor(3)
        assert result["addr"] == "10.0.0.1"
        # local_rank 3, port 29523 + 100 = 29623
        assert result["nccl_port"] == 29623

    def test_no_donor_returns_none(self):
        """No config returns None."""
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.nccl_loader import NCCLWeightLoader

        loader = NCCLWeightLoader(LoadConfig(
            load_format="nccl",
            model_loader_extra_config={},
        ))
        assert loader._discover_donor(0) is None
