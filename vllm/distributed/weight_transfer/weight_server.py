# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL P2P weight server daemon thread.

Spawned inside each TP worker after ``load_model()``. Serves the worker's
model parameters to new vLLM instances via NCCL, reading directly from
existing ``nn.Parameter`` GPU tensors (zero additional GPU memory in
steady state).

Protocol:
    1. Receiver TCP-connects to ``control_port`` (base_port + tp_rank).
    2. Server sends length-prefixed JSON:
       ``{status, tp_rank, node_index, num_tensors, total_bytes, nccl_port}``.
    3. Receiver sends ``"ACK\\n"``.
    4. Both create ``StatelessProcessGroup`` on ``nccl_port``.
    5. Server broadcasts metadata via ``pg.broadcast_obj()`` (packed v2).
    6. Server packs and broadcasts tensors via NCCL (2 GB buffers).
    7. Both tear down NCCL group.
    8. Server returns to listening.

Port layout (default base_port=29520, TP=8):
    Control ports: 29520–29527 (TCP probe/handshake)
    NCCL ports:    29620–29627 (StatelessProcessGroup)
"""

import contextlib
import json
import socket
import struct
import threading
import time
from collections.abc import Callable

import torch
import torch.nn as nn

from vllm.distributed.weight_transfer.packed_buffer import (
    DEFAULT_BUFFER_SIZE,
    compute_buffer_plan,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_NCCL_PORT_OFFSET = 100


class WeightServer:
    """NCCL P2P weight server daemon thread.

    Serves the worker's model parameters to new vLLM instances via NCCL.
    Reads directly from existing ``nn.Parameter`` GPU tensors — zero
    additional GPU memory in steady state. Transfers are serialized
    (one receiver at a time) since a single transfer saturates the
    IB port (~47 GB/s per link).

    Args:
        model: The loaded ``nn.Module`` whose parameters to serve.
        device: CUDA device index.
        tp_rank: Tensor-parallel rank (used for port computation).
        port: Base port for control channels.
        is_available_fn: Callable returning ``True`` when model is
            available for weight serving (``False`` during sleep, etc.).
        node_index: Node index for multi-node TP discovery metadata.
        buffer_size: Packed buffer size in bytes for NCCL transfer.
        nccl_port_offset: Offset from control port to NCCL port.
    """

    def __init__(
        self,
        model: nn.Module,
        device: int,
        tp_rank: int,
        port: int,
        is_available_fn: Callable[[], bool],
        node_index: int = 0,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        nccl_port_offset: int = _DEFAULT_NCCL_PORT_OFFSET,
    ) -> None:
        self.model = model
        self.device = device
        self.tp_rank = tp_rank
        self.control_port = port + tp_rank
        self.nccl_port = port + nccl_port_offset + tp_rank
        self.is_available_fn = is_available_fn
        self.node_index = node_index
        self.buffer_size = buffer_size
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn daemon thread."""
        self._thread = threading.Thread(
            target=self._serve_loop,
            daemon=True,
            name=f"weight-server-{self.tp_rank}",
        )
        self._thread.start()
        logger.info(
            "WeightServer: tp_rank=%d listening on port %d (NCCL: %d)",
            self.tp_rank, self.control_port, self.nccl_port,
        )

    def stop(self) -> None:
        """Signal shutdown (daemon thread exits with process anyway)."""
        self._shutdown.set()

    def _serve_loop(self) -> None:
        """Accept connections and serve weights one at a time."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.control_port))
        server.listen(5)
        server.settimeout(1.0)

        while not self._shutdown.is_set():
            try:
                conn, addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            try:
                self._handle_connection(conn, addr)
            except Exception:
                logger.exception(
                    "WeightServer: error serving %s", addr,
                )
            finally:
                with contextlib.suppress(OSError):
                    conn.close()

        server.close()

    def _handle_connection(
        self, conn: socket.socket, addr: tuple
    ) -> None:
        """Handle a single weight transfer request."""
        logger.info("WeightServer: connection from %s", addr)

        if not self.is_available_fn():
            self._send_json(conn, {"status": "unavailable"})
            logger.info("WeightServer: rejected (model not available)")
            return

        # Build tensor metadata from model (zero-copy references)
        state_dict = dict(self.model.state_dict())
        tensor_metas: list[tuple[str, list[int], str, int]] = []
        total_bytes = 0
        for name, tensor in state_dict.items():
            nbytes = tensor.numel() * tensor.element_size()
            tensor_metas.append(
                (name, list(tensor.shape), str(tensor.dtype), nbytes)
            )
            total_bytes += nbytes

        # Send metadata summary via TCP
        self._send_json(conn, {
            "status": "ready",
            "tp_rank": self.tp_rank,
            "node_index": self.node_index,
            "num_tensors": len(tensor_metas),
            "total_bytes": total_bytes,
            "nccl_port": self.nccl_port,
        })

        # Wait for ACK
        ack = conn.recv(4)
        if not ack.startswith(b"ACK"):
            logger.warning("WeightServer: no ACK received, aborting")
            return

        # NCCL transfer
        self._transfer_weights(state_dict, tensor_metas, total_bytes)

    def _transfer_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        tensor_metas: list[tuple[str, list[int], str, int]],
        total_bytes: int,
    ) -> None:
        """Pack and broadcast weights via NCCL."""
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from vllm.distributed.utils import StatelessProcessGroup

        t0 = time.monotonic()
        pg = StatelessProcessGroup.create(
            host="0.0.0.0", port=self.nccl_port, rank=0, world_size=2,
        )
        nccl = PyNcclCommunicator(pg, device=self.device)
        logger.info(
            "WeightServer: NCCL rendezvous in %.3fs",
            time.monotonic() - t0,
        )

        buffer = None
        try:
            # Send full metadata (packed v2 format)
            pg.broadcast_obj({
                "version": 2,
                "tensors": tensor_metas,
                "buffer_size": self.buffer_size,
                "format": "state_dict",
            }, src=0)

            # Pack and broadcast on dedicated CUDA stream
            plan = compute_buffer_plan(tensor_metas, self.buffer_size)
            max_buf = max((bb for bb, _ in plan), default=0)
            buffer = torch.empty(
                max(max_buf, 1), dtype=torch.uint8, device=self.device,
            )

            nccl_stream = torch.cuda.Stream(self.device)
            t_transfer = time.monotonic()

            with torch.cuda.stream(nccl_stream):
                for _buf_idx, (buf_bytes, assignments) in enumerate(plan):
                    for tensor_idx, buf_offset in assignments:
                        name = tensor_metas[tensor_idx][0]
                        tensor = state_dict[name]
                        # .reshape(-1) handles 0-dim scalar tensors
                        flat = tensor.contiguous().reshape(-1).view(
                            torch.uint8
                        )
                        buffer[buf_offset:buf_offset + flat.numel()] = flat
                    nccl.broadcast(buffer[:buf_bytes], src=0)

            nccl_stream.synchronize()
            elapsed = time.monotonic() - t_transfer
            total_gb = total_bytes / 1024**3
            throughput = total_gb / elapsed if elapsed > 0 else 0
            logger.info(
                "WeightServer: served %.1f GB in %.3fs (%.1f GB/s), "
                "%d tensors in %d buffers",
                total_gb, elapsed, throughput,
                len(tensor_metas), len(plan),
            )
        finally:
            del buffer, nccl, pg

    @staticmethod
    def _send_json(conn: socket.socket, data: dict) -> None:
        """Send length-prefixed JSON over TCP."""
        payload = json.dumps(data).encode("utf-8")
        length = struct.pack("!I", len(payload))
        conn.sendall(length + payload)
