# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight server daemon."""

import json
import socket
import struct


class TestWeightServerConfig:
    """Test WeightServer configuration (no GPU required)."""

    def test_port_computation(self):
        """Control and NCCL ports computed from base + tp_rank."""
        # Use a mock model
        import torch.nn as nn

        from vllm.distributed.weight_transfer.weight_server import (
            WeightServer,
        )
        model = nn.Linear(4, 4)

        server = WeightServer(
            model=model,
            device=0,
            tp_rank=3,
            port=29520,
            is_available_fn=lambda: True,
            node_index=1,
        )
        assert server.control_port == 29523
        assert server.nccl_port == 29623
        assert server.node_index == 1

    def test_custom_port_offset(self):
        """Custom nccl_port_offset changes NCCL port."""
        import torch.nn as nn

        from vllm.distributed.weight_transfer.weight_server import (
            WeightServer,
        )
        model = nn.Linear(4, 4)

        server = WeightServer(
            model=model,
            device=0,
            tp_rank=0,
            port=29520,
            is_available_fn=lambda: True,
            nccl_port_offset=200,
        )
        assert server.control_port == 29520
        assert server.nccl_port == 29720

    def test_custom_buffer_size(self):
        """Custom buffer_size is stored."""
        import torch.nn as nn

        from vllm.distributed.weight_transfer.weight_server import (
            WeightServer,
        )
        model = nn.Linear(4, 4)

        server = WeightServer(
            model=model,
            device=0,
            tp_rank=0,
            port=29520,
            is_available_fn=lambda: True,
            buffer_size=1024**3,
        )
        assert server.buffer_size == 1024**3


class TestSendJson:
    """Test the TCP protocol helper."""

    def test_send_json_format(self):
        """_send_json sends length-prefixed JSON."""
        from vllm.distributed.weight_transfer.weight_server import (
            WeightServer,
        )

        # Create connected socket pair
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", port))
        conn, _ = server_sock.accept()

        try:
            data = {"status": "ready", "tp_rank": 0, "nccl_port": 29620}
            WeightServer._send_json(conn, data)

            # Read length prefix
            length_bytes = client_sock.recv(4)
            length = struct.unpack("!I", length_bytes)[0]

            # Read payload
            payload = client_sock.recv(length)
            received = json.loads(payload.decode("utf-8"))

            assert received == data
        finally:
            conn.close()
            client_sock.close()
            server_sock.close()
