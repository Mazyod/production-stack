import os
import sys
import tempfile

import yaml

from vllm_router.dynamic_config import DynamicRouterConfig
from vllm_router.parsers import parser


def test_dynamic_router_config_timeout_keep_alive_defaults_to_5():
    cfg = DynamicRouterConfig(service_discovery="static", routing_logic="roundrobin")
    assert cfg.timeout_keep_alive == 5


def test_from_args_carries_timeout_keep_alive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            sys.argv[0],
            "--service-discovery",
            "static",
            "--static-backends",
            "http://localhost:8000",
            "--static-models",
            "m1",
            "--routing-logic",
            "roundrobin",
            "--timeout-keep-alive",
            "120",
        ],
    )
    args = parser.parse_args()
    cfg = DynamicRouterConfig.from_args(args)
    assert cfg.timeout_keep_alive == 120


def test_from_yaml_accepts_timeout_keep_alive_without_crashing():
    # Regression: a config file carrying timeout_keep_alive must not make the
    # watcher's strict DynamicRouterConfig(**config) reject the unknown key.
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(
                {
                    "service_discovery": "static",
                    "routing_logic": "roundrobin",
                    "timeout_keep_alive": 120,
                },
                f,
            )
        cfg = DynamicRouterConfig.from_yaml(path)
    finally:
        os.unlink(path)
    assert cfg.timeout_keep_alive == 120
