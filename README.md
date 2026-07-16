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
- **Structured-output boundary repair**: `--enable-structured-output-repair` (off by default) repairs vLLM's grammar-constrained JSON when speculative decoding + thinking corrupt the reasoning→answer boundary (`` ```json{ ``, `{{`, `{"{`). The router locates the true document using the caller's own JSON Schema as an oracle and only commits when the output is syntactically impossible as a truncation; on any doubt it returns the backend's original bytes, byte-for-byte. See [Structured-output boundary repair](#structured-output-boundary-repair) below.
- **Request-stats lifecycle fix**: In-flight counters no longer drift upward forever. `on_request_complete` sat outside a `finally`, so a client disconnect (`GeneratorExit`/`CancelledError` — both `BaseException`, so `except Exception` missed them) skipped the decrement and every abandoned request leaked its stage count for the life of the process; the per-request timestamp dicts were never popped even on success, growing without bound. `RequestStatsMonitor` now hands out an opaque per-attempt handle and exposes idempotent `on_request_complete` / `on_request_fail` / `on_request_abort`, retiring each attempt from the stage its own record says it is in. Backend sockets are bounded to match: `--backend-connect-timeout` (10 s) and `--backend-read-timeout` (300 s of silence, not of duration) stop a black-holed engine from hanging a request — and its counters — forever. See [Request-stats lifecycle](#request-stats-lifecycle) below.

Pre-built images are published to Docker Hub, tagged to match upstream releases:

```console
docker pull openimage/production-stack-router:v0.1.10
```

### Audio-enabled vLLM serving image

The stock `vllm/vllm-openai` image ships **without** the audio extras, so `POST /v1/audio/transcriptions` fails at request time with `ImportError: Please install vllm[audio] for audio support` (surfaced to clients as a generic `400 Invalid or unsupported audio file`). This fork also publishes a drop-in replacement that installs the vLLM `audio` extra (`av`, `soundfile`, `soxr`, `scipy`, … — the set tracks the vLLM version), pinned to the exact vLLM build in the base image so nothing else changes:

```console
docker pull openimage/vllm-openai-audio:v0.10.0
```

It tracks vLLM core releases (a separate cadence from the router), is built from `docker/Dockerfile.audio` by the [`build-vllm-audio`](.github/workflows/build-vllm-audio.yml) workflow, and keeps the upstream entrypoint — swap the image and Whisper/transcription endpoints just work.

### Structured-output boundary repair

`--enable-structured-output-repair` enables router-side repair for content generated under a discriminating **object-rooted** JSON Schema. It is off by default. Requests using `logprobs`, schema-less `json_object` mode, **array- or scalar-rooted** schemas, non-discriminating schemas, tool schemas alone, and non-2xx backend responses remain on the existing byte-preserving path.

Buffering defaults to 1 MiB and 30 seconds; configure it with `--structured-output-repair-max-bytes` and `--structured-output-repair-max-seconds`. Any cap, timeout, ambiguity, exception, or transport failure replays the original bytes.

Diagnostic captures for `ambiguous` and `unknown` outcomes are disabled by default. Enable them with `--structured-output-repair-capture-dir` only after creating a router-owned `0700` directory. Captures are sampled, structurally redacted, capped at 4 KiB, retained for seven days, and written as `0600` files. Raw model output is never placed in metric labels or ordinary logs.

See the [structured-output boundary-repair design](docs/superpowers/specs/2026-07-13-structured-output-boundary-repair-design.md) for safety properties and limits.

This is a narrow safety net for responses where a short garbage prefix precedes a complete JSON document. The caller's schema is used to locate and validate that document. A schema is discriminating only when it has a non-empty `required` list or sets `additionalProperties` to `false`. Scalar roots, array roots, `json_object` mode, non-discriminating schemas, requests with `logprobs`, and non-2xx responses do not engage repair. Tool-call repair is out of scope in this version.

For an engaged streaming response, the router buffers from the first content frame until the stream ends. This increases time to first token in exchange for correctness: a structured-output caller needs the complete document before parsing it. On any doubt, the router returns the original response bytes byte-for-byte. Only a `repaired` outcome changes the body; in every other outcome, enabling the feature changes no response bytes.

The operator flags and their defaults are:

- `--enable-structured-output-repair`: disabled.
- `--structured-output-repair-max-bytes`: `1048576` bytes (1 MiB), the per-request streaming buffer cap.
- `--structured-output-repair-max-seconds`: `30`, the buffering deadline in seconds.
- `--structured-output-repair-capture-dir`: unset, so diagnostic capture is disabled.
- `--structured-output-repair-capture-sample-rate`: `0.01`, the fraction of `ambiguous` and `unknown` outcomes captured.
- `--structured-output-repair-capture-max-bytes`: `4096`, the maximum UTF-8 size of a redacted output excerpt.
- `--structured-output-repair-capture-retention-days`: `7` days.

The capture sink writes at most 64 MiB in total across all retained capture files, independent of the per-record capture limit. The configured directory must be owned by the router user and have mode `0700`; capture files use mode `0600`.

The feature exports these Prometheus metrics:

- `vllm:structured_output_repairs_total{model,status,mode}` counts engaged repair outcomes.
- `vllm:structured_output_garbage_prefix_bytes{model}` measures the UTF-8 byte length removed by successful repairs.
- `vllm:structured_output_schema_rejections_total{model,reason}` counts schemas rejected at the engagement boundary.

Use the `status` label on `vllm:structured_output_repairs_total` as follows:

- `repaired`: a corrupt response was fixed. This is the only status where the client received a changed body.
- `clean`: repair engaged, but nothing was wrong.
- `incomplete`: the backend reported `finish_reason: "length"`, so the router declined repair.
- `ambiguous` or `unknown`: the router could not safely repair the output. Investigate these outcomes; enable the capture sink to inspect redacted samples.
- `no_terminal`: the stream ended with a content index that never received a `finish_reason`.
- `poisoned`: a frame arrived that the router could not prove was content-free, so it replayed the buffered response.
- `capped`: the byte cap was reached. Consider whether `--structured-output-repair-max-bytes` is too low for the workload.
- `timeout`: the buffering deadline was reached, which can indicate a slow backend.
- `error`: an exception occurred on the transform path and should be investigated.

For every status except `repaired`, the client receives the original bytes unchanged.

#### Golden-corpus regression test

`src/tests/test_structured_output_corpus.py` validates repair behavior against `matrix_results.json`, a corpus of 1,536 real production requests. That file is not included in this repository, and the test skips when `STRUCTURED_OUTPUT_CORPUS` is unset. Run it with:

```bash
STRUCTURED_OUTPUT_CORPUS=/path/to/matrix_results.json uv run pytest src/tests/test_structured_output_corpus.py -q
```

The field mapping at the top of the test module is unverified against the real corpus. If the mapping is wrong, the harness fails loudly with mapping or classification errors; it never silently passes.

#### Related semantic-cache behavior changes

Two routing changes affect chat-completion traffic even when structured-output repair is disabled:

1. Semantic-cache lookup now uses the post-rewrite request body. It previously ran before routing against the pre-rewrite body. This is currently inert because only `NoopRequestRewriter` ships, but enabling a body-changing rewriter in the future will change cache keys for all chat-completion traffic, independently of the repair flag.
2. `callbacks.pre_request(...)` now runs on semantic-cache hits. Cache hits were previously returned from `route_chat_completion` before reaching `route_general_request`, so callbacks did not run for cache-served responses. This means authentication, quota, and logging callbacks can now observe cache hits.

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

On breach, the attempt fails: if response headers have not arrived yet, the failover loop retries the request on the next engine; after headers (including a stream that stalls before its first body byte) the client's stream terminates instead of hanging, since a response is already committed. Either way the counters retire through the ordinary `fail` path, and the error is visible as `request_errors_total{error_type="ConnectionTimeoutError"|"SocketTimeoutError"|"TimeoutError"}`.

The one behavior change: a non-streaming generation — headers arrive only when generation finishes — or a deeply queued request whose backend stays silent longer than 300 s is now terminated and retried on another engine rather than waiting indefinitely. Raise `--backend-read-timeout` (or set `0`) if your workloads legitimately stay silent longer; prefer streaming for very long generations.

Note that `vllm:num_requests_running` is exported by the router under the same metric name vLLM engines export natively. Dashboards and autoscaler queries should filter by `job` or component to avoid selecting the router-derived series. The stock autoscalers are unaffected: the router HPA scales on CPU, and the engine KEDA trigger uses `vllm:num_requests_waiting` from the engine scraper.

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
