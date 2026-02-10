# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL-based model weight loader for P2P weight transfer.

Receives model weights from a running vLLM weight server (or external
cache process) via NCCL, enabling fast scale-from-zero startup. Each TP
worker independently discovers and connects to its paired donor GPU.

Supports two modes:
- Broadcast: all workers receive all weights, each ``load_weights()``
  handles TP/PP/EP sharding internally (simpler, more bandwidth).
- Sharded (P2P): each worker receives only its pre-sharded slice
  via a dedicated 2-rank NCCL group (less bandwidth, zero redundancy).

Discovery protocol (tried in order):
1. ``model_loader_extra_config["nccl_addrs"]`` — comma-separated donor IPs
2. ``model_loader_extra_config["nccl_addr"]`` — single donor IP
3. ``model_loader_extra_config["service_name"]`` — K8s headless service DNS
4. Fallback to ``DefaultModelLoader`` (disk loading)
"""

import json
import os
import socket
import struct
import time

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed.weight_transfer.packed_buffer import (
    DEFAULT_BUFFER_SIZE,
    compute_buffer_plan,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

# Default port offsets and timeouts
_DEFAULT_NCCL_PORT_OFFSET = 100
_DEFAULT_DONOR_PORT = 29520
_DEFAULT_PROBE_TIMEOUT = 5.0  # seconds
_DEFAULT_NCCL_TIMEOUT = 300  # seconds


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a dtype string like ``'torch.float16'`` to ``torch.dtype``."""
    attr = dtype_str.split(".")[-1]
    return getattr(torch, attr, torch.uint8)


@register_model_loader("nccl")
class NCCLWeightLoader(BaseModelLoader):
    """Load model weights via NCCL from a weight donor.

    Configuration via ``model_loader_extra_config``:

    - ``nccl_addr`` (str): Single donor IP address.
    - ``nccl_addrs`` (str): Comma-separated donor IPs for multi-node TP.
    - ``nccl_port`` (int): Donor base port (default 29520).
    - ``nccl_port_offset`` (int): Offset from control to NCCL port (default 100).
    - ``gpus_per_node`` (int): GPUs per node for multi-node routing (default 8).
    - ``mode`` (str): ``"broadcast"`` or ``"sharded"`` (default ``"sharded"``).
    - ``buffer_size`` (int): Packed buffer size in bytes (default 2 GB).
    - ``timeout`` (int): NCCL transfer timeout in seconds (default 300).
    - ``service_name`` (str): K8s headless service for DNS discovery.
    """

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}

        # Discovery config
        self._addr: str = extra.get("nccl_addr", "")
        addrs_str = extra.get("nccl_addrs", "")
        self._addrs: list[str] = [
            a.strip() for a in addrs_str.split(",") if a.strip()
        ] if addrs_str else []
        self._port: int = int(extra.get("nccl_port", str(_DEFAULT_DONOR_PORT)))
        self._port_offset: int = int(
            extra.get("nccl_port_offset", str(_DEFAULT_NCCL_PORT_OFFSET))
        )
        self._gpus_per_node: int = int(extra.get("gpus_per_node", "8"))
        self._service_name: str = extra.get("service_name", "")

        # Transfer config
        self._mode: str = extra.get("mode", "sharded")
        self._buffer_size: int = int(
            extra.get("buffer_size", str(DEFAULT_BUFFER_SIZE))
        )
        self._timeout: int = int(
            extra.get("timeout", str(_DEFAULT_NCCL_TIMEOUT))
        )

        self._tp_size: int = 1

    def download_model(self, model_config: ModelConfig) -> None:
        """No download needed — weights come via NCCL or disk fallback."""

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        """Capture parallel config before ``load_weights()`` runs."""
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        return super().load_model(vllm_config, model_config, prefix)

    def load_weights(
        self, model: nn.Module, model_config: ModelConfig
    ) -> None:
        """Discover donor and load weights via NCCL P2P."""
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from vllm.distributed.utils import StatelessProcessGroup

        tp_rank = get_tensor_model_parallel_rank()
        device = torch.cuda.current_device()

        donor_info = self._discover_donor(tp_rank)
        if donor_info is None:
            logger.info(
                "NCCLWeightLoader: no donor found, falling back to disk"
            )
            self._fallback_to_default(model, model_config)
            return

        addr = donor_info["addr"]
        nccl_port = donor_info["nccl_port"]
        logger.info(
            "NCCLWeightLoader: tp_rank=%d, donor=%s:%d, mode=%s",
            tp_rank, addr, nccl_port, self._mode,
        )

        if self._mode == "sharded":
            self._load_sharded(
                model, addr, nccl_port, device,
                StatelessProcessGroup, PyNcclCommunicator,
            )
        else:
            raise ValueError(
                f"NCCLWeightLoader: unsupported mode '{self._mode}'. "
                f"Use 'sharded'."
            )

    def _load_sharded(
        self,
        model: nn.Module,
        addr: str,
        nccl_port: int,
        device: int,
        StatelessProcessGroup: type,
        PyNcclCommunicator: type,
    ) -> None:
        """Receive pre-sharded state dict via 2-rank P2P NCCL group."""
        t0 = time.monotonic()
        pg = StatelessProcessGroup.create(
            host=addr, port=nccl_port, rank=1, world_size=2,
        )
        nccl = PyNcclCommunicator(pg, device=device)
        logger.info(
            "NCCLWeightLoader: rendezvous in %.3fs",
            time.monotonic() - t0,
        )

        # Receive packed metadata
        metadata = pg.broadcast_obj(None, src=0)
        tensor_metas = metadata["tensors"]
        buffer_size = metadata["buffer_size"]
        plan = compute_buffer_plan(tensor_metas, buffer_size)
        logger.info(
            "NCCLWeightLoader: %d tensors in %d buffers",
            len(tensor_metas), len(plan),
        )

        # Build parameter mapping for in-place copy
        param_map: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            param_map[name] = param
        for name, buf in model.named_buffers():
            param_map[name] = buf

        # Receive and copy in-place
        t0 = time.monotonic()
        total_bytes = 0
        assigned = 0
        fallback_count = 0

        for buf_bytes, assignments in plan:
            buf = torch.empty(buf_bytes, dtype=torch.uint8, device=device)
            nccl.broadcast(buf, src=0)
            for tensor_idx, buf_offset in assignments:
                name, shape, dtype_str, nbytes = tensor_metas[tensor_idx]
                dtype = _resolve_dtype(dtype_str)
                tensor = buf[buf_offset:buf_offset + nbytes].view(
                    dtype
                ).reshape(shape)
                if name in param_map:
                    param_map[name].data.copy_(tensor)
                    assigned += 1
                else:
                    if self._assign_by_path(model, name, tensor):
                        fallback_count += 1
                    else:
                        logger.warning(
                            "NCCLWeightLoader: failed to assign %s", name
                        )
                total_bytes += nbytes

        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - t0
        total_gb = total_bytes / 1024**3
        throughput = total_gb / elapsed if elapsed > 0 else 0
        logger.info(
            "NCCLWeightLoader: %.1f GB in %.3fs (%.1f GB/s), "
            "%d direct, %d fallback",
            total_gb, elapsed, throughput, assigned, fallback_count,
        )

        del nccl, pg

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_donor(self, tp_rank: int) -> dict | None:
        """Multi-tier donor discovery.

        For multi-node TP, routes each tp_rank to the correct donor pod
        based on ``gpus_per_node``.

        Returns:
            Dict with ``addr`` and ``nccl_port`` keys, or ``None``.
        """
        local_tp_rank = tp_rank % self._gpus_per_node
        control_port = self._port + local_tp_rank

        # Tier 0: Multi-address (TP across pods)
        if self._addrs:
            pod_index = tp_rank // self._gpus_per_node
            if pod_index < len(self._addrs):
                addr = self._addrs[pod_index]
                info = self._probe(addr, control_port)
                if info:
                    return {"addr": addr, **info}
                logger.warning(
                    "NCCLWeightLoader: donor pod %d (%s:%d) not reachable",
                    pod_index, addr, control_port,
                )

        # Tier 1: Single address
        if self._addr:
            info = self._probe(self._addr, control_port)
            if info:
                return {"addr": self._addr, **info}
            logger.warning(
                "NCCLWeightLoader: donor %s:%d not reachable",
                self._addr, control_port,
            )

        # Tier 2: K8s headless service DNS
        if self._service_name:
            try:
                addrs = socket.getaddrinfo(
                    self._service_name, None, socket.AF_INET,
                )
                for _, _, _, _, (ip, _) in addrs:
                    info = self._probe(ip, control_port)
                    if info:
                        return {"addr": ip, **info}
            except socket.gaierror:
                logger.warning(
                    "NCCLWeightLoader: DNS lookup failed for %s",
                    self._service_name,
                )

        return None

    def _probe(
        self, addr: str, port: int, timeout: float = _DEFAULT_PROBE_TIMEOUT
    ) -> dict | None:
        """TCP probe to verify weight server readiness.

        Returns metadata dict (including ``nccl_port``) or ``None``.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((addr, port))
            # Length-prefixed JSON
            length_bytes = self._recv_exact(sock, 4)
            length = struct.unpack("!I", length_bytes)[0]
            payload = self._recv_exact(sock, length)
            metadata = json.loads(payload.decode("utf-8"))

            if metadata.get("status") != "ready":
                logger.info(
                    "NCCLWeightLoader: donor %s:%d status=%s",
                    addr, port, metadata.get("status"),
                )
                return None

            sock.sendall(b"ACK\n")
            return metadata
        except (TimeoutError, OSError, ConnectionRefusedError, ValueError):
            return None
        finally:
            import contextlib
            with contextlib.suppress(OSError):
                sock.close()

    # ------------------------------------------------------------------
    # Fallback & helpers
    # ------------------------------------------------------------------

    def _fallback_to_default(
        self, model: nn.Module, model_config: ModelConfig
    ) -> None:
        """Fall back to ``DefaultModelLoader`` when no donor is available."""
        import glob as _glob

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        fallback_config = LoadConfig(
            load_format="auto",
            download_dir=self.load_config.download_dir,
            model_loader_extra_config={},
        )

        model_path = model_config.model
        has_weights = bool(
            _glob.glob(os.path.join(model_path, "*.safetensors"))
            or _glob.glob(os.path.join(model_path, "*.bin"))
        )
        if not has_weights:
            repo_id = model_path
            if "models--" in model_path:
                repo_part = model_path.split("models--")[1]
                repo_part = repo_part.split("/snapshots/")[0]
                repo_id = repo_part.replace("--", "/", 1)
            logger.info(
                "NCCLWeightLoader: no weights at %s, downloading %s...",
                model_path, repo_id,
            )
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id)
            object.__setattr__(model_config, "model", model_path)

        DefaultModelLoader(fallback_config).load_weights(model, model_config)

    @staticmethod
    def _assign_by_path(
        model: nn.Module, name: str, tensor: torch.Tensor
    ) -> bool:
        """Assign tensor to model parameter by dotted path.

        Returns:
            ``True`` if assignment succeeded, ``False`` otherwise.
        """
        parts = name.split(".")
        module = model
        try:
            for p in parts[:-1]:
                module = getattr(module, p)
            target = getattr(module, parts[-1], None)
            if target is not None and isinstance(
                target, (torch.Tensor, nn.Parameter)
            ):
                target.data.copy_(tensor)
                return True
        except (AttributeError, TypeError) as e:
            logger.warning(
                "NCCLWeightLoader: _assign_by_path failed for %s: %s",
                name, e,
            )
        return False

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        """Receive exactly *n* bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ValueError("Connection closed")
            data += chunk
        return data
