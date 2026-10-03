# Multi-stage build: fat builder with uv, slim runtime with just the venv.

# ── Builder ───────────────────────────────────────────────────
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .
COPY .git/ .git/
COPY src/ src/

ARG INSTALL_OPTIONAL_DEP=
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir setuptools_scm && \
    uv pip install --no-cache-dir .[$INSTALL_OPTIONAL_DEP]

# ── Runtime ───────────────────────────────────────────────────
FROM python:3.13-slim

COPY --from=builder /opt/venv /opt/venv

ENTRYPOINT ["/opt/venv/bin/vllm-router"]
CMD []
