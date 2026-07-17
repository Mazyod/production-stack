import json
import os
import sys
import tempfile

import yaml

from vllm_router.dynamic_config import DynamicRouterConfig
from vllm_router.parsers import parser

# The fork's startup-only operator flags. They are honored at startup (via
# parser.set_defaults) but are NOT reconfigurable fields, so the dynamic-config
# watcher must accept and ignore them rather than crash on the strict
# DynamicRouterConfig(**config) construction.
FORK_STARTUP_ONLY_FLAGS = {
    "backend_connect_timeout": 20.0,
    "backend_read_timeout": 600.0,
}


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


def test_from_yaml_tolerates_fork_and_upstream_startup_only_keys():
    # A config file may carry the fork's startup-only flags plus arbitrary
    # upstream flags. The watcher's strict DynamicRouterConfig(**config) must
    # not crash on them; they are dropped (honored at startup, not hot-reloaded)
    # while the reconfigurable fields still parse.
    document = {
        "service_discovery": "static",
        "routing_logic": "roundrobin",
        # a non-reconfigurable upstream flag
        "engine_stats_interval": 45,
        **FORK_STARTUP_ONLY_FLAGS,
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(document, f)
        cfg = DynamicRouterConfig.from_yaml(path)
    finally:
        os.unlink(path)

    assert cfg.service_discovery == "static"
    assert cfg.routing_logic == "roundrobin"
    # The dropped keys must not have leaked onto the dataclass.
    for key in {*FORK_STARTUP_ONLY_FLAGS, "engine_stats_interval"}:
        assert not hasattr(cfg, key), f"{key} should have been filtered out"


def test_from_json_tolerates_fork_and_upstream_startup_only_keys():
    document = {
        "service_discovery": "static",
        "routing_logic": "roundrobin",
        "engine_stats_interval": 45,
        **FORK_STARTUP_ONLY_FLAGS,
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(document, f)
        cfg = DynamicRouterConfig.from_json(path)
    finally:
        os.unlink(path)

    assert cfg.service_discovery == "static"
    assert cfg.routing_logic == "roundrobin"
    for key in {*FORK_STARTUP_ONLY_FLAGS, "engine_stats_interval"}:
        assert not hasattr(cfg, key), f"{key} should have been filtered out"


def test_fork_flags_are_inert_to_the_watcher():
    # Two config files identical except that one carries the fork's
    # startup-only flags must yield equal DynamicRouterConfig objects, proving
    # the flags cannot perturb the watcher's `config != current_config` check
    # and thus cannot trigger a spurious reconfigure.
    base = {"service_discovery": "external-only", "routing_logic": "roundrobin"}
    with_flags = {**base, **FORK_STARTUP_ONLY_FLAGS}

    paths = []
    try:
        cfgs = []
        for document in (base, with_flags):
            fd, path = tempfile.mkstemp(suffix=".yaml")
            paths.append(path)
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(document, f)
            cfgs.append(DynamicRouterConfig.from_yaml(path))
    finally:
        for path in paths:
            os.unlink(path)

    assert cfgs[0] == cfgs[1]
