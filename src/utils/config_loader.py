"""Validated YAML configuration with environment-variable overrides.

Defaults come from ``config/default_config.yaml`` (single source of truth). Layers, lowest to
highest priority: defaults < user YAML file < ``TRAFFIC__SECTION__KEY=value`` environment
variables (values are parsed as YAML scalars, so ``true``, ``4.5`` and ``[1, 2]`` work).
Unknown keys / override paths and out-of-range values raise ``ConfigError``.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from ..errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"
ENV_PREFIX = "TRAFFIC__"

# path -> (type(s), min, max, choices)
_RULES: Dict[str, tuple] = {
    "system.log_level": (str, None, None, {"DEBUG", "INFO", "WARNING", "ERROR"}),
    "perception.backend": (str, None, None, {"torch", "onnx"}),
    "perception.device": (str, None, None, None),
    "perception.confidence_threshold": ((int, float), 0.0, 1.0, None),
    "perception.nms_threshold": ((int, float), 0.0, 1.0, None),
    "perception.input_size": (int, 32, 4096, None),
    "perception.max_stale_s": ((int, float), 0.0, 3600.0, None),
    "perception.max_consecutive_failures": (int, 1, 1000, None),
    "tracking.max_age": (int, 1, 10000, None),
    "tracking.min_hits": (int, 1, 100, None),
    "tracking.iou_threshold": ((int, float), 0.0, 1.0, None),
    "tracking.output_coasting": (int, 0, 10000, None),
    "queue_estimation.stopped_speed": ((int, float), 1e-6, 100.0, None),
    "emergency.detection_confidence": ((int, float), 0.0, 1.0, None),
    "emergency.min_detections": (int, 1, 100, None),
    "signal.decision_interval_s": ((int, float), 0.1, 60.0, None),
    "signal.policy": (str, None, None, {"fixed", "actuated", "rl"}),
    "rl.dqn.gamma": ((int, float), 0.0, 1.0, None),
    "rl.dqn.learning_rate": ((int, float), 1e-9, 1.0, None),
    "simulation.step_length_s": ((int, float), 0.01, 5.0, None),
    "simulation.episode_s": ((int, float), 1.0, 1e7, None),
}


def _merge(base: Dict[str, Any], over: Mapping[str, Any], path: str = "") -> None:
    for k, v in over.items():
        if k not in base:
            raise ConfigError(f"unknown configuration key {path + str(k)!r}")
        if isinstance(base[k], dict) and isinstance(v, Mapping):
            _merge(base[k], v, f"{path}{k}.")
        else:
            base[k] = v


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> None:
        with open(DEFAULT_PATH, encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)
        self.config_path = config_path
        if config_path:
            p = Path(config_path)
            if not p.is_file():
                raise ConfigError(f"configuration file not found: {p}")
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
            if not isinstance(data, Mapping):
                raise ConfigError(f"{p}: top level must be a mapping")
            _merge(self.config, data)
        self._apply_env(os.environ if env is None else env)
        self.validate()

    def _apply_env(self, env: Mapping[str, str]) -> None:
        for key, raw in env.items():
            if not key.startswith(ENV_PREFIX):
                continue
            parts = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
            node: Any = self.config
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ConfigError(f"environment override {key}: unknown path")
                node = node[part]
            if not isinstance(node, dict) or parts[-1] not in node:
                raise ConfigError(f"environment override {key}: unknown key")
            try:
                node[parts[-1]] = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ConfigError(f"environment override {key}: bad value {raw!r}") from exc

    def validate(self) -> None:
        for path, (types, lo, hi, choices) in _RULES.items():
            val = self.get(path)
            if val is None:
                raise ConfigError(f"missing required setting {path}")
            if isinstance(val, bool) or not isinstance(val, types):
                raise ConfigError(f"{path}: expected {types}, got {type(val).__name__} ({val!r})")
            if (lo is not None and val < lo) or (hi is not None and val > hi):
                raise ConfigError(f"{path}={val!r} outside [{lo}, {hi}]")
            if choices is not None and val not in choices:
                raise ConfigError(f"{path}={val!r} not one of {sorted(choices)}")
        from ..safety.signal_safety import SignalTiming  # deeper timing checks live with the type

        SignalTiming.from_dict(self.config["signal"]["timing"])

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self.config
        for part in key.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def get_section(self, section: str) -> Dict[str, Any]:
        return copy.deepcopy(self.config.get(section, {}))

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.config)
