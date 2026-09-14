---
name: verify
description: Launch and drive the vllm_router end-to-end against a fake engine to verify router changes at the HTTP surface.
---

# Verifying router changes end-to-end

Run from the repository root after the setup in `CONTRIBUTING.md`.
For regression coverage, run `.venv/bin/python -m pytest src/tests -q`.
For behavior changes, also exercise the actual router process as below.

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
- Router log: connect failures include `Backend connect failure` and `attempt=i/n`; read/entry timeouts report the bound and return 504 without rotating.
- Metrics surface: `curl -s http://127.0.0.1:18800/metrics | grep -E "num_requests_running|request_errors_total"` — in-flight counters must drain to 0 after every request; errors land in `vllm:request_errors_total{error_type=...}`.
- Use a closed localhost port for connection refusal. A connect-timeout test needs a controlled peer/network that drops traffic; an arbitrary private IP may refuse immediately or reach a real service.
- For timeout changes, cover a stall before headers (504), a mid-stream SSE stall (terminal error + `[DONE]`), and a healthy stream that outlasts the read bound while continuing to emit bytes. Enable failover explicitly when testing connect failures and 502 exhaustion.
- For config changes, set the value only in YAML/JSON and verify its effect at the HTTP surface. Observe an unchanged file across watcher ticks, then check that a typo is rejected while the running config stays intact.

## Gotchas

- Poll `/health` with a bounded wait until HTTP 200 before sending scenarios; startup is not instantaneous.
- Use temporary files and record process IDs. Stop only the router and fake engine processes you started, including on failure.
- Report the scenarios, observed statuses/events, and counter cleanup. Do not label a unit-only run or a health response as full inference verification.
