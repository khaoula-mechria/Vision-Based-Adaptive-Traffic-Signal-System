"""Run policies through the same environment and report measured metrics."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from ..control.policies import SignalPolicy
from ..rl_agent.traffic_env import TrafficSignalEnv


def run_policy_episode(env: TrafficSignalEnv, policy: SignalPolicy, seed: int) -> Dict[str, float]:
    """One episode with ``policy`` choosing a phase request every ``env.decision_interval_s``."""
    env.reset(seed=seed)
    policy.reset()
    total, done = 0.0, False
    while not done:
        assert env.intersection_obs is not None
        req = policy.request(env.intersection_obs, env.context())
        _, r, terminated, truncated, _ = env.step(req.phase)
        total += r
        done = terminated or truncated
    out = env.summary()
    out["return"] = total
    return out


def compare_policies(
    env_factories: Mapping[str, Callable[[], TrafficSignalEnv]],
    policies: Mapping[str, SignalPolicy],
    seeds: Sequence[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate each named policy on its own env (so decision intervals may differ) over the
    same seeds. Returns ``{policy: {metric: {mean, std, per_seed}}}``."""
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, policy in policies.items():
        env = env_factories[name]()
        try:
            runs: List[Dict[str, float]] = [run_policy_episode(env, policy, s) for s in seeds]
        finally:
            env.close()
        results[name] = {
            k: {
                "mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs])),
                "per_seed": [float(r[k]) for r in runs],
            }
            for k in runs[0]
        }
    return results
