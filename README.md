# vLLM Production Stack (fork)

## About this fork

This is a lightweight fork of [vllm-project/production-stack](https://github.com/vllm-project/production-stack) with the following changes:

- **Multi-stage Docker build**: Builder stage installs dependencies with uv, runtime stage is just `python:3.13-slim` with the venv copied over. Image size reduced from ~5 GB to ~294 MB by defaulting `INSTALL_OPTIONAL_DEP` to empty (no PyTorch/sentence-transformers/vLLM).
- **Python 3.13**: Both builder and runtime stages use Python 3.13.
- **Single-label hostname fix**: `validate_url` regex accepts Docker/K8s hostnames without dots (e.g., `http://vllm-worker:8000`). Mirrors [PR #737](https://github.com/vllm-project/production-stack/pull/737).
- **Qwen rerank template**: Preprocesses `/v1/rerank` requests for `Qwen/Qwen3-Reranker-0.6B` with the required chat template before forwarding to the backend.
- **`/pooling` route**: Proxies vLLM's `/pooling` endpoint (backend chosen by the request body's `model` field, same as `/v1/embeddings`). Needed for Jina Embeddings v4 multi-vector (ColBERT) output, which vLLM serves only on `/pooling`.
- **Default port 8080**: Changed from 8001 to match the [vllm-project/router](https://github.com/vllm-project/router) default for easier future migration.
- **numpy unpinned**: `>=1.26.4` instead of `==1.26.4` (no Python 3.13 wheels for 1.26.4).
- **Request-stats lifecycle fix**: In-flight counters no longer drift upward forever. `on_request_complete` sat outside a `finally`, so a client disconnect (`GeneratorExit`/`CancelledError` — both `BaseException`, so `except Exception` missed them) skipped the decrement and every abandoned request leaked its stage count for the life of the process; the per-request timestamp dicts were never popped even on success, growing without bound. `RequestStatsMonitor` now hands out an opaque per-attempt handle and exposes idempotent `on_request_complete` / `on_request_fail` / `on_request_abort`, retiring each attempt from the stage its own record says it is in. Backend sockets are bounded to match: `--backend-connect-timeout` (10 s) and `--backend-read-timeout` (300 s of silence, not of duration) stop a black-holed engine from hanging a request — and its counters — forever, and timeouts surface as structured **504/502** responses with an OpenAI-style error envelope (SSE streams get an in-band terminal error event) instead of bare 500s. See [Request-stats lifecycle](#request-stats-lifecycle) below.
- **Dynamic-config file accepts all fork flags**: The `--dynamic-config-yaml` / `--dynamic-config-json` watcher used to reject any key that was not a hot-reloadable field, raising `TypeError` every 10 s and silently killing hot-reload of service discovery / routing. The fork's startup-only flags — the backend socket timeouts — therefore could not live in the config file you already pass, even though they load fine at startup. The watcher now tolerates non-reconfigurable keys: they are honored at startup and ignored (debug-logged) on reload, so you can keep all configuration in one file. See [Loading fork flags from the dynamic config file](#loading-fork-flags-from-the-dynamic-config-file) below.

Pre-built images are published to Docker Hub, tagged to match upstream releases:

```console
docker pull openimage/production-stack-router:v0.1.10
```

### Audio-enabled vLLM serving image

The stock `vllm/vllm-openai` image ships **without** the audio extras, so `POST /v1/audio/transcriptions` fails at request time with `ImportError: Please install vllm[audio] for audio support` (surfaced to clients as a generic `400 Invalid or unsupported audio file`). A drop-in replacement installs the vLLM `audio` extra (`av`, `soundfile`, `soxr`, `scipy`, … — the set tracks the vLLM version), pinned to the exact vLLM build in the base image so nothing else changes:

```console
docker pull openimage/vllm-openai-audio:v0.25.1
```

> **This image now lives in the vLLM engine fork, [`Mazyod/vllm`](https://github.com/Mazyod/vllm).** Everything about the vLLM **engine** was consolidated there; production-stack owns only the **router**. The fork layers the audio extra **and** a small series of upstream bugfix backports onto a pinned `vllm/vllm-openai` release — see its [`FORK.md`](https://github.com/Mazyod/vllm/blob/main/FORK.md). The image name, registry, and drop-in entrypoint are unchanged, so `openimage/vllm-openai-audio` stays the pull target and swapping the image keeps Whisper/transcription endpoints working.

### Request-stats lifecycle

`RequestStatsMonitor` tracks in-flight requests as per-engine counters split by stage: `on_new_request` increments `in_prefill_requests`, the first response token moves the count to `in_decoding_requests`, and completion retires it. Upstream, the completion hook sits outside a `finally`, so any terminal path that is not a clean return skips it.

The counters are per-engine aggregates rather than per-request, so adding a `finally` alone is not sufficient and is actively harmful: retiring a request that never reached decoding decrements `in_decoding_requests` anyway, stealing the decrement owed to a *different* concurrent request while the aborted request's prefill count leaks regardless. Upstream's multipart path demonstrates this — it already has the `finally` and still corrupts counters.

The monitor now issues an opaque handle per backend attempt. The external `X-Request-Id` is caller-controlled and two concurrent requests may share one, so it is retained as metadata only and never used as identity. Each attempt has one authoritative active record naming the stage it is in; the finalizer pops that record first and retires the attempt from its recorded stage, which makes repeated or unknown finalization a no-op. All state and snapshots are guarded by a `threading.RLock`, because `--log-stats` runs a real OS thread that reads the same sliding-window buffers the event loop mutates.

Terminal outcomes are explicit: `on_request_complete` for normal backend exhaustion, `on_request_fail` for a backend or transport error, and `on_request_abort` for client disconnect or cancellation.

Behavior changes worth noting:

- `finished_requests` now counts normal exhaustion only. Aborted and failed attempts no longer increment it. A response whose body is read to completion counts as exhaustion even when its status is 4xx or 5xx.
- Abandoned streams are recorded as `status="aborted"` in `vllm:request_latency_seconds`. They were previously recorded as `success`, because the status defaulted to success and cancellation bypassed the error handler.
- A request body that is not JSON-parsable now returns 400. It previously raised `TypeError` and surfaced as a 500, because `HTTPException(status=...)` is an invalid keyword — the parameter is `status_code`.
- Failed failover attempts now retire. Each attempt owns its own handle and finalizes in its own `finally`, so an attempt that fails against one engine no longer leaves a count behind on it.

#### Backend socket timeouts

The lifecycle fix guarantees that a terminating attempt retires its counters; socket timeouts guarantee that attempts terminate. Proxied backend requests previously used `ClientTimeout(total=None)` — no bound of any kind — so an engine that vanished without closing the connection (node death, TCP black hole, dropped SYNs) hung the request and its stage count forever, and the failover loop never ran because it only sees failures, not silence.

Backend requests now default to `--backend-connect-timeout` 10 s and `--backend-read-timeout` 300 s (`0` disables either). `total` stays `None`: the read bound is on *silence*, not duration — it re-arms on every byte received, so a stream that keeps producing tokens can run for hours, and aiohttp suspends the watchdog while the router applies backpressure to a slow client, so a slow reader cannot trip it either. Every silent phase of an attempt is covered: DNS resolution and connection establishment by the connect bound (`ClientTimeout.connect` + `sock_connect`), the wait for response headers and gaps mid-stream by `sock_read`, and the request-body upload — where `sock_read` is not yet armed, so a backend that accepts the connection but stops reading would otherwise hang a large multimodal payload forever — by an entry deadline of connect + read (310 s by default) that is disarmed the moment response headers arrive.

On breach, the client gets a structured answer instead of a bare 500 or a silent connection reset:

- **Connect-phase failures** (`ConnectionTimeoutError`, connection refused) rotate to the next engine; when every permitted failover attempt has failed, the router returns **502** with an OpenAI-style envelope — `{"error": {"message": …, "type": "bad_gateway", "code": "backend_connect_error", "param": null}}`.
- **Read/entry timeouts before response headers** return **504** immediately — `code` is `backend_read_timeout` (read gap or header wait) or `backend_entry_timeout` (connect + upload + header deadline), `type` is `gateway_timeout`, and the message carries the configured bound. They deliberately do **not** rotate backends: the stall is workload-shaped, and a retry would typically eat the same bound on the next engine, multiplying worst-case latency.
- **Mid-stream stalls on SSE responses** (status already committed) end with an in-band terminal event — `data: {"error": {…, "code": "backend_stream_stall"}}` followed by `data: [DONE]` — and a clean close, mirroring the engine's own streaming error contract. Non-SSE bodies keep the abrupt close: a truncated body must not be dressed up as success.

Every error response carries `X-Request-Id` and `Retry-After: 1`. Each timeout logs one structured `WARNING` (request id, backend, model, endpoint, elapsed, which bound fired) with no traceback, the counters retire through the ordinary `fail` path, and the error stays visible as `request_errors_total{error_type="ConnectionTimeoutError"|"SocketTimeoutError"|"TimeoutError"}`.

The one behavior change: a non-streaming generation — headers arrive only when generation finishes — or a deeply queued request whose backend stays silent longer than 300 s is now terminated with the 504 above rather than waiting indefinitely. Raise `--backend-read-timeout` (or set `0`) if your workloads legitimately stay silent longer; prefer streaming for very long generations.

Note that `vllm:num_requests_running` is exported by the router under the same metric name vLLM engines export natively. Dashboards and autoscaler queries should filter by `job` or component to avoid selecting the router-derived series. The stock autoscalers are unaffected: the router HPA scales on CPU, and the engine KEDA trigger uses `vllm:num_requests_waiting` from the engine scraper.

### Loading fork flags from the dynamic config file

If you launch the router with `--dynamic-config-yaml` (or `--dynamic-config-json`), that file is your single source of truth: you do not also need to spell the fork's flags out in the `command:` array. Every flag the router accepts — including the fork's startup-only ones below — can be set as a key in that file.

The startup-only fork flags and their config keys:

| Config key | Flag | Default |
|---|---|---|
| `backend_connect_timeout` | `--backend-connect-timeout` | `10.0` |
| `backend_read_timeout` | `--backend-read-timeout` | `300.0` |

Example:

```yaml
service_discovery: static
routing_logic: roundrobin
static_models:
  my-model:
    static_backends:
      - http://vllm-worker:8000

# Fork flags — same file, no command-line flags needed:
backend_connect_timeout: 10.0
backend_read_timeout: 300.0
```

Three things to know:

- **Keys use underscores, not dashes.** The config key is `backend_read_timeout`, matching the flag's destination — not `backend-read-timeout`. A dashed key matches no flag and is silently ignored.
- **These flags are read once, at startup.** Editing them in the file while the router runs has no effect until you restart it. Only the hot-reloadable fields (`static_backends`, `routing_logic`, `callbacks`, …) take effect on a live edit; the watcher re-reads the file every 10 s but applies only that subset. The startup-only flags are inert to the watcher, so editing one is a no-op (logged at `debug`), not a reconfigure.
- **A key that is not a recognized flag is rejected, and your running config is kept.** If the watcher re-reads the file and finds a key that is neither a hot-reloadable field nor a known flag — almost always a typo (e.g. `callback` instead of `callbacks`) — it logs a warning and does **not** reconfigure, so the running configuration is preserved rather than silently reverting the mistyped field to its default. Fix the typo and the next reload applies cleanly.

---

<!-- markdownlint-disable-next-line MD025 -->
# vLLM Production Stack: reference stack for production vLLM deployment

| [**Blog**](https://lmcache.github.io) | [**Docs**](https://docs.vllm.ai/projects/production-stack) | [**Production-Stack Slack Channel**](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack) | [**LMCache Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ) | [**Interest Form**](https://forms.gle/mQfQDUXbKfp2St1z7) |

## Latest News

- 📄 [Official documentation](https://docs.vllm.ai/projects/production-stack) released for production-stack!
- ✨ [Cloud Deployment Tutorials](https://github.com/vllm-project/production-stack/blob/main/tutorials) for Lambda Labs, AWS EKS, Google GCP are out!
- 🛤️ 2026 roadmap is released! [Join the discussion now](https://github.com/vllm-project/production-stack/issues/855)!
- 🔥 vLLM Production Stack is released! Check out our [release blogs](https://blog.lmcache.ai/2025-01-21-stack-release) posted on January 22, 2025.

## Community Events

We host **bi-weekly** community meetings at the following timeslot:

- Every other Tuesdays at 5:30 PM PT – [Add to Calendar](https://drive.google.com/uc?export=download&id=1D4SqQiqzdSx_xsEwS0QTd592zd3Xourh)

All are welcome to join!

## Introduction

**vLLM Production Stack** project provides a reference implementation on how to build an inference stack on top of vLLM, which allows you to:

- 🚀 Scale from a single vLLM instance to a distributed vLLM deployment without changing any application code
- 💻 Monitor the metrics through a web dashboard
- 😄 Enjoy the performance benefits brought by request routing and KV cache offloading

## Step-By-Step Tutorials

0. How To [*Install Kubernetes (kubectl, helm, minikube, etc)*](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md)?
1. How to [*Deploy Production Stack on Major Cloud Platforms (AWS, GCP, Lambda Labs, Azure)*](https://github.com/vllm-project/production-stack/blob/main/tutorials/cloud_deployments)?
2. How To [*Set up a Minimal vLLM Production Stack*](https://github.com/vllm-project/production-stack/blob/main/tutorials/01-minimal-helm-installation.md)?
3. How To [*Customize vLLM Configs (optional)*](https://github.com/vllm-project/production-stack/blob/main/tutorials/02-basic-vllm-config.md)?
4. How to [*Load Your LLM Weights*](https://github.com/vllm-project/production-stack/blob/main/tutorials/03-load-model-from-pv.md)?
5. How to [*Launch Different LLMs in vLLM Production Stack*](https://github.com/vllm-project/production-stack/blob/main/tutorials/04-launch-multiple-model.md)?
6. How to [*Enable KV Cache Offloading with LMCache*](https://github.com/vllm-project/production-stack/blob/main/tutorials/05-offload-kv-cache.md)?

## Architecture

The stack is set up using [Helm](https://helm.sh/docs/), and contains the following key parts:

- **Serving engine**: The vLLM engines that run different LLMs.
- **Request router**: Directs requests to appropriate backends based on routing keys or session IDs to maximize KV cache reuse.
- **Observability stack**: monitors the metrics of the backends through [Prometheus](https://github.com/prometheus/prometheus) + [Grafana](https://grafana.com/)

<p align="center">
  <img src="https://github.com/user-attachments/assets/8f05e7b9-0513-40a9-9ba9-2d3acca77c0c" alt="Architecture of the stack" width="80%"/>
</p>

## Roadmap

We are actively working on this project and will release the following features soon. Please stay tuned!

- **Autoscaling** based on vLLM-specific metrics
- Support for **disaggregated prefill**
- **Router improvements** (e.g., more performant router using non-python languages, KV-cache-aware routing algorithm, better fault tolerance, etc)

## Deploying the stack via Helm

### Prerequisites

- A running Kubernetes (K8s) environment with GPUs
  - Run `cd utils && bash install-minikube-cluster.sh`
  - Or follow our [tutorial](tutorials/00-install-kubernetes-env.md)

### Deployment

vLLM Production Stack can be deployed via helm charts. Clone the repo to local and execute the following commands for a minimal deployment:

```bash
git clone https://github.com/vllm-project/production-stack.git
cd production-stack/
helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f tutorials/assets/values-01-minimal-example.yaml
```

The deployed stack provides the same [**OpenAI API interface**](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?ref=blog.mozilla.ai#openai-compatible-server) as vLLM, and can be accessed through kubernetes service.

To validate the installation and send a query to the stack, refer to [this tutorial](tutorials/01-minimal-helm-installation.md).

For more information about customizing the helm chart, please refer to [values.yaml](https://github.com/vllm-project/production-stack/blob/main/helm/values.yaml) and our other [tutorials](https://github.com/vllm-project/production-stack/tree/main/tutorials).

### Uninstall

```bash
helm uninstall vllm
```

## Grafana Dashboard

### Features

The Grafana dashboard provides the following insights:

1. **Available vLLM Instances**: Displays the number of healthy instances.
2. **Request Latency Distribution**: Visualizes end-to-end request latency.
3. **Time-to-First-Token (TTFT) Distribution**: Monitors response times for token generation.
4. **Number of Running Requests**: Tracks the number of active requests per instance.
5. **Number of Pending Requests**: Tracks requests waiting to be processed.
6. **GPU KV Usage Percent**: Monitors GPU KV cache usage.
7. **GPU KV Cache Hit Rate**: Displays the hit rate for the GPU KV cache.

<p align="center">
  <img src="https://github.com/user-attachments/assets/05766673-c449-4094-bdc8-dea6ac28cb79" alt="Grafana dashboard to monitor the deployment" width="80%"/>
</p>

### Configuration

See the details in [`helm/README.md`](./helm/README.md#Observability)

## Router

The router ensures efficient request distribution among backends. It supports:

- Routing to endpoints that run different models
- Exporting observability metrics for each serving engine instance, including QPS, time-to-first-token (TTFT), number of pending/running/finished requests, and uptime
- Automatic service discovery and fault tolerance via the Kubernetes API
- Model aliases
- Multiple routing algorithms:
  - Round-robin routing
  - Session-ID based routing
  - Prefix-aware routing (WIP)

Please refer to the [router documentation](./src/vllm_router/README.md) for more details.

## Contributing

We welcome and value any contributions and collaborations. Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

## License

This project is licensed under Apache License 2.0. See the `LICENSE` file for details.

## Sponsors

We are grateful to our sponsors who support our development and benchmarking efforts:

<p align="center">
  <a href="https://gmicloud.ai">
    <img src="https://cdn.prod.website-files.com/6683d8c52e4e62685a8d90cf/67a0a0064683945b0cf77f25_GMI%20Cloud%20Logo_Black.svg" alt="GMI Cloud Logo" width="200"/>
  </a>
</p>

---

For any issues or questions, feel free to open an issue or contact us ([@ruizhang0101](https://github.com/ruizhang0101), [@ApostaC](https://github.com/ApostaC), [@YuhanLiu11](https://github.com/YuhanLiu11), [@Shaoting-Feng](https://github.com/Shaoting-Feng)).
