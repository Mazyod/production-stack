---
name: verify
description: Launch and drive the vllm_router end-to-end against a fake engine to verify router changes at the HTTP surface.
---

# Verifying router changes end-to-end

## Launch

The router entry point is `.venv/bin/vllm-router` (`vllm_router.app:main`). Minimal viable flags:

```bash
.venv/bin/vllm-router --port 18800 \
  --routing-logic roundrobin --service-discovery static \
  --static-backends http://127.0.0.1:18801 --static-models test-model
```

Multiple backends: comma-separate both `--static-backends` and `--static-models` (one model entry per backend). Failover is OFF by default; enable with `--max-instance-failover-reroute-attempts N`.

## Fake engine

A tiny aiohttp server standing in for vLLM only needs `GET /v1/models` (return `{"object":"list","data":[{"id":"test-model","object":"model"}]}`) and `POST /v1/chat/completions`. Script the completion handler per scenario (stall forever, stream slowly, etc.). The router's engine-stats scraper will spam 404s for `/metrics` — harmless noise.

## Observe

- Response surface: `curl -s -w "\nHTTP %{http_code} in %{time_total}s\n"` (add `-N` for streams, `-m N` to cap a deliberate hang).
- Router log: failover shows as `WARNING ... failed on <url> (attempt i/n): <error>`.
- Metrics surface: `curl -s :18800/metrics | grep -E "num_requests_running|request_errors_total"` — in-flight counters must drain to 0 after every request; errors land in `vllm:request_errors_total{error_type=...}`.
- A black-holed backend is simulated well by `http://10.255.255.1:9` (unroutable, drops SYNs) — exercises connect timeouts and failover.

## Gotchas

- Startup takes ~2-3s before the port answers; sleep before curling.
- Kill the router and fake engine PIDs when done; they run detached.
