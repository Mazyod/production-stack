# vLLM Production Stack (fork)

## About this fork

This is a lightweight fork of [vllm-project/production-stack](https://github.com/vllm-project/production-stack) with the following changes:

- **Multi-stage Docker build**: Builder stage installs dependencies with uv, runtime stage is just `python:3.13-slim` with the venv copied over. The default image omits PyTorch, sentence-transformers, and vLLM. Proposed upstream as a slim image variant in [issue #1238](https://github.com/vllm-project/production-stack/issues/1238).
- **Python 3.13**: Both builder and runtime stages use Python 3.13.
- **Single-label hostname fix**: `validate_url` regex accepts Docker/K8s hostnames without dots (e.g., `http://vllm-worker:8000`). Upstream fix pending in [PR #1154](https://github.com/vllm-project/production-stack/pull/1154), which supersedes the closed #737.
- **Qwen rerank template**: Preprocesses `/v1/rerank` requests for `Qwen/Qwen3-Reranker-0.6B` with the required chat template before forwarding to the backend. Malformed or non-object JSON uses upstream's shared 400 response contract.
- **Default port 8080**: Changed from 8001 to match the [vllm-project/router](https://github.com/vllm-project/router) default for easier future migration.
- **Model alias discovery**: `/v1/models` lists static aliases alongside canonical model names so clients can discover the names they may request. The per-request `Routing request …` log line ends with `requested_model=` (the name the client sent), `model=` (after alias resolution), `stream=` and `max_tokens=` (`max_completion_tokens` when set; `None` when the client set no limit). Proposed upstream in [issue #1237](https://github.com/vllm-project/production-stack/issues/1237).
- **Request lifecycle and backend timeouts**: Each attempt records completion, failure, or client cancellation exactly once. Connect and read-silence bounds return structured 502/504 errors, including an in-band terminal event for stalled SSE streams. See [Request-stats lifecycle](#request-stats-lifecycle). Upstream: the aborted latency status is [PR #1234](https://github.com/vllm-project/production-stack/pull/1234); the timeout findings are on [PR #1082](https://github.com/vllm-project/production-stack/pull/1082#issuecomment-7133124388).
- **Early client disconnects abort the engine request**: While it waits for backend response headers, the router watches the client the way vLLM's `@with_cancellation` does. vLLM sends a non-streaming response's headers only when generation finishes, so a client that gave up on a non-streaming request used to leave the engine generating until completion or `--backend-read-timeout`. The router now closes the upstream request at once and the engine aborts it. The attempt retires as an abort and logs `Client disconnected before headers` with its request ID and elapsed time. Disconnects after headers, including streams, behave as before. This covers the JSON routes such as chat, completions and embeddings; `/v1/rerank` (which replays a rewritten body), multipart audio and image routes, and disaggregated prefill do not watch the client before headers. Upstream: we support the earlier equivalent [PR #1040](https://github.com/vllm-project/production-stack/pull/1040) for [issue #634](https://github.com/vllm-project/production-stack/issues/634) and closed our duplicate #1233.
- **Multipart compatibility**: Includes upstream audio translation routing, text/SRT/VTT responses, and automatic transcription language detection. Priority and prefix routing also work with multipart requests. HTML backend error pages and malformed backend JSON still return 502, with the fork's lifecycle cleanup preserved.
- **Backend DNS recovery**: Backend and external-provider hostnames are resolved with the system resolver (`getaddrinfo`) instead of aiohttp's c-ares default, which `aiohttp[speedups]` enables. c-ares kept its own cache of failed lookups inside the router for up to an hour, so a backend Service or container created after the router first tried it kept returning 502 `backend_connect_error` until that entry expired or the router restarted. The router now asks the system resolver again on the next request. Negative caching in CoreDNS, systemd-resolved, or another upstream resolver still applies, and aiohttp still caches successful lookups for 10 s. This covers the proxy client, disaggregated-prefill sessions, sleep/wake calls, and OpenAI-compatible external providers; running routers need the new image to pick it up. The shared-client change is pending in [upstream PR #1162](https://github.com/vllm-project/production-stack/pull/1162). See [Backend DNS resolution](docs/fork-maintenance.md#backend-dns-resolution--october-2026).
- **Dynamic-config file accepts all fork flags**: The `--dynamic-config-yaml` / `--dynamic-config-json` watcher used to reject any key that was not a hot-reloadable field, raising `TypeError` every 10 s and silently killing hot-reload of service discovery / routing. The fork's startup-only flags — the backend socket timeouts — therefore could not live in the config file you already pass, even though they load fine at startup. The watcher now tolerates non-reconfigurable keys: they are honored at startup and ignored (debug-logged) on reload, so you can keep all configuration in one file. See [Loading fork flags from the dynamic config file](#loading-fork-flags-from-the-dynamic-config-file) below. Pending upstream in [PR #1235](https://github.com/vllm-project/production-stack/pull/1235).

The checkout includes upstream `vllm-stack-0.1.14` at `405a2fa`. Upstream now
provides `POST /pooling`, the uvicorn keep-alive flag, per-dispatch identity,
and Python 3.13-compatible NumPy. Ordinary streams are forwarded without
retaining the full body. A configured response callback receives the complete
body for requests dispatched while it is active. The fork also preserves live
discovery reload for static backends that omit model types.

The published router image is available for linux/amd64 and linux/arm64. See the [0.1.14 release record](docs/fork-maintenance.md#authorized-release--2026-11-24) for the verified digest.

```console
docker pull openimage/production-stack-router:v0.1.14
```

The release workflow applies the reconciled fork changes, including merge resolutions, onto an upstream release tag and runs the router regression suite before publishing versioned images. It requires HTTP 200 from the image's `/health` before promoting `latest`.

For development, start with [AGENTS.md](AGENTS.md) and the [fork maintenance guide](docs/fork-maintenance.md), which records the release workflow and decisions shared by coding tools.

### Audio-enabled vLLM serving image

The stock `vllm/vllm-openai` image ships **without** the audio extras, so `POST /v1/audio/transcriptions` fails at request time with `ImportError: Please install vllm[audio] for audio support` (surfaced to clients as a generic `400 Invalid or unsupported audio file`). A drop-in replacement installs the vLLM `audio` extra (`av`, `soundfile`, `soxr`, `scipy`, … — the set tracks the vLLM version), pinned to the exact vLLM build in the base image so nothing else changes:

```console
docker pull openimage/vllm-openai-audio:v0.25.1
```

> **This image now lives in the vLLM engine fork, [`Mazyod/vllm`](https://github.com/Mazyod/vllm).** Everything about the vLLM **engine** was consolidated there; production-stack owns only the **router**. The fork layers the audio extra **and** a small series of upstream bugfix backports onto a pinned `vllm/vllm-openai` release — see its [`FORK.md`](https://github.com/Mazyod/vllm/blob/main/FORK.md). The image name, registry, and drop-in entrypoint are unchanged, so `openimage/vllm-openai-audio` stays the pull target and swapping the image keeps Whisper/transcription endpoints working.

### Request-stats lifecycle

`RequestStatsMonitor` tracks in-flight requests as per-engine counters split by stage: `on_new_request` increments `in_prefill_requests`, the first response token moves the count to `in_decoding_requests`, and completion retires it. At the original fork baseline, the completion hook sat outside a `finally`, so terminal paths other than a clean return skipped it. Upstream 0.1.14 includes stage-aware cleanup and a unique ID per dispatch. The fork also tracks the terminal outcome of each attempt, locks concurrent statistics reads, and bounds backend silence.

The counters are per-engine aggregates rather than per-request, so adding a `finally` alone is not sufficient and is actively harmful: retiring a request that never reached decoding decrements `in_decoding_requests` anyway, stealing the decrement owed to a *different* concurrent request while the aborted request's prefill count leaks regardless. The fork therefore preserves stage-aware finalization alongside the upstream cleanup fix.

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

A non-streaming generation — headers arrive only when generation finishes — or a deeply queued request whose backend stays silent longer than 300 s is now terminated with the 504 above rather than waiting indefinitely. Raise `--backend-read-timeout` (or set `0`) if your workloads legitimately stay silent longer; prefer streaming for very long generations.

Note that `vllm:num_requests_running` is exported by the router under the same metric name vLLM engines export natively. Dashboards and autoscaler queries should filter by `job` or component to avoid selecting the router-derived series. The stock autoscalers are unaffected: the router HPA scales on CPU, and the engine KEDA trigger uses `vllm:num_requests_waiting` from the engine scraper.

### Loading fork flags from the dynamic config file

If you launch the router with `--dynamic-config-yaml` (or `--dynamic-config-json`), that file is your single source of truth: you do not also need to spell the fork's flags out in the `command:` array. Every flag the router accepts — including the fork's startup-only ones below — can be set as a key in that file.

The backend timeouts and upstream keep-alive setting take these config keys:

| Config key | Flag | Default |
|---|---|---|
| `backend_connect_timeout` | `--backend-connect-timeout` | `10.0` |
| `backend_read_timeout` | `--backend-read-timeout` | `300.0` |
| `timeout_keep_alive` | `--timeout-keep-alive` | `5` |

Example:

```yaml
service_discovery: static
routing_logic: roundrobin
static_models:
  my-model:
    static_backends:
      - http://vllm-worker:8000

# Startup-only settings
backend_connect_timeout: 10.0
backend_read_timeout: 300.0
timeout_keep_alive: 5
```

Three things to know:

- **Keys use underscores, not dashes.** The config key is `backend_read_timeout`, matching the flag's destination — not `backend-read-timeout`. A dashed key has no effect at startup and is rejected by the watcher on reload.
- **These flags are read once, at startup.** Restart to change their effective values. Backend timeout edits are inert to the watcher (logged at `debug`). `timeout_keep_alive` is a legacy dataclass field: editing it triggers a reconfiguration but does not change uvicorn's keep-alive timeout. Only the hot-reloadable settings (`static_backends`, `routing_logic`, `callbacks`, …) take effect on a live edit; the watcher re-reads the file every 10 seconds.
- **A key that is not a recognized flag is rejected, and your running config is kept.** If the watcher re-reads the file and finds a key that is neither a hot-reloadable field nor a known flag — almost always a typo (e.g. `callback` instead of `callbacks`) — it logs a warning and does **not** reconfigure, so the running configuration is preserved rather than silently reverting the mistyped field to its default. Fix the typo and the next reload applies cleanly.

---

<!-- markdownlint-disable-next-line MD025 -->
# vLLM Production Stack: reference stack for production vLLM deployment

| [**Blog**](https://lmcache.github.io) | [**Docs**](https://docs.vllm.ai/projects/production-stack) | [**Production-Stack Slack Channel**](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack) | [**LMCache Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ) | [**Interest Form**](https://forms.gle/mQfQDUXbKfp2St1z7) |

<p align="center">
  <b>English</b> · <a href="README.zh.md">简体中文</a>
</p>

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
