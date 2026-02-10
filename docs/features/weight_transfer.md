# NCCL Weight Transfer

vLLM supports transferring model weights between running instances via NCCL P2P, enabling fast scale-from-zero startup. A running vLLM instance (the *donor*) serves its model parameters to new instances (the *receivers*) over InfiniBand or NVLink — zero additional GPU memory on the donor side.

Key benefits:

- **Fast startup**: New instances receive weights via NCCL at near line-rate (40+ GB/s per IB link), bypassing disk I/O entirely.
- **Zero extra GPU memory**: The donor reads directly from its existing model parameters.
- **Automatic fallback**: If no donor is reachable, the receiver falls back to loading from disk.
- **Multi-node TP**: Supports tensor parallelism across multiple nodes, with each TP worker independently connecting to its paired donor GPU.

!!! note
    This feature requires CUDA and NCCL. The donor and receiver must be on nodes with network connectivity (InfiniBand or TCP).

## Architecture

```
Donor Pod (running vLLM + weight server)
  Worker 0 (GPU 0) ──NCCL P2P──> Receiver Worker 0
  Worker 1 (GPU 1) ──NCCL P2P──> Receiver Worker 1
  ...
  Worker 7 (GPU 7) ──NCCL P2P──> Receiver Worker 7
```

Each donor TP worker runs a daemon thread that:

1. Listens on a TCP control port for incoming requests.
2. On connection, sends metadata (tensor count, total bytes, NCCL port).
3. Creates a 2-rank NCCL group with the receiver.
4. Packs and broadcasts model parameters via NCCL.

## Quick Start

### Donor (existing instance)

Start vLLM with the weight server enabled:

```bash
VLLM_WEIGHT_SERVER_ENABLED=1 \
VLLM_WEIGHT_SERVER_PORT=29520 \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8
```

### Receiver (new instance)

Start vLLM with the NCCL loader, pointing to the donor:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 8 \
  --load-format nccl \
  --model-loader-extra-config '{"nccl_addr": "10.0.0.1"}'
```

If the donor is unreachable, the receiver automatically falls back to loading from disk.

## Configuration

### Donor Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_WEIGHT_SERVER_ENABLED` | `0` | Set to `1` to enable the weight server daemon. |
| `VLLM_WEIGHT_SERVER_PORT` | `29520` | Base port for control channels. Control ports use `base + tp_rank`, NCCL ports use `base + 100 + tp_rank`. |

### Receiver Configuration

The receiver uses `--load-format nccl` with options in `--model-loader-extra-config`:

| Key | Default | Description |
|-----|---------|-------------|
| `nccl_addr` | `""` | Single donor IP address. |
| `nccl_addrs` | `""` | Comma-separated donor IPs for multi-node TP (e.g., `"10.0.0.1,10.0.0.2"`). |
| `nccl_port` | `29520` | Donor base port. |
| `gpus_per_node` | `8` | GPUs per node (for multi-node TP rank routing). |
| `mode` | `"sharded"` | Transfer mode. Use `"sharded"` for pre-sharded P2P. |
| `buffer_size` | `2147483648` | Packed buffer size in bytes (default 2 GB). |
| `timeout` | `300` | NCCL transfer timeout in seconds. |
| `service_name` | `""` | Kubernetes headless service name for DNS-based donor discovery. |

### Discovery Protocol

The receiver discovers the donor through a multi-tier protocol:

1. **Multi-address** (`nccl_addrs`): Routes each TP rank to the correct donor pod based on `gpus_per_node`.
2. **Single address** (`nccl_addr`): Connects all ranks to a single donor.
3. **Kubernetes DNS** (`service_name`): Resolves a headless service to discover donor IPs.
4. **Fallback**: If no donor is found, loads from disk using `DefaultModelLoader`.

### Port Layout

With base port 29520 and TP=8:

```
Control ports (TCP): 29520-29527
NCCL ports (P2P):    29620-29627
```

For multi-node TP=16 across 2 nodes, each node uses the same port range since they have different IPs.

## Multi-Node TP Example

For TP=16 across 2 nodes (8 GPUs each):

**Donor nodes:**

```bash
# Both donor nodes use the same command
VLLM_WEIGHT_SERVER_ENABLED=1 \
  vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 16
```

**Receiver nodes:**

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 16 \
  --load-format nccl \
  --model-loader-extra-config '{"nccl_addrs": "10.0.0.1,10.0.0.2", "gpus_per_node": "8"}'
```

TP ranks 0-7 connect to the first donor (10.0.0.1), ranks 8-15 connect to the second (10.0.0.2).

## Kubernetes Deployment

For Kubernetes, use a headless service for automatic discovery:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-weight-server
spec:
  clusterIP: None
  selector:
    app: vllm-donor
  ports:
    - name: control-0
      port: 29520
      targetPort: 29520
```

Then configure the receiver with `service_name`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 8 \
  --load-format nccl \
  --model-loader-extra-config '{"service_name": "vllm-weight-server"}'
```
