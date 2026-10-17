#!/usr/bin/env python
"""Compare fixed-time, actuated and DQN control on identical SUMO demand (same seeds).

    python scripts/evaluate.py --checkpoint runs/dqn_sumo/best.pt --scenarios medium high --seeds 9000 9001 9002

All three controllers go through the same SignalSafetyManager; only the phase *requests* differ.
Baselines decide every 1 s, the RL policy every ``--rl-interval`` s. Results are written as
JSON + Markdown; nothing here is hard-coded or estimated - every number is measured in SUMO.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from src.control import ActuatedPolicy, FixedTimePolicy, RLPolicy  # noqa: E402
from src.evaluation import compare_policies  # noqa: E402
from src.rl_agent import ObservationEncoder, TrafficSignalEnv  # noqa: E402
from src.safety import IntersectionModel, SignalTiming  # noqa: E402
from src.simulation import SensorNoise, SumoBackend, sumo_available  # noqa: E402

COLS = [
    ("mean_vehicle_wait_s", "veh wait mean [s]"), ("max_vehicle_wait_s", "veh wait max [s]"),
    ("mean_pedestrian_wait_s", "ped wait mean [s]"), ("mean_queue", "queue"),
    ("throughput_veh", "throughput [veh]"), ("mean_travel_time_s", "travel [s]"),
    ("phase_switches", "switches"), ("mean_emergency_delay_s", "emerg. delay [s]"),
    ("collisions", "collisions"), ("safety_violations", "audit viol."), ("ped_conflict_steps", "ped-conflict steps"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="DQN checkpoint (omit to compare baselines only)")
    ap.add_argument("--scenarios", nargs="+", default=["low", "medium", "high", "rush_hour", "asymmetric", "ped_heavy", "emergency"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[9000, 9001, 9002, 9003, 9004])
    ap.add_argument("--episode-s", type=float, default=1800.0)
    ap.add_argument("--rl-interval", type=float, default=5.0)
    ap.add_argument("--ped-detect-prob", type=float, default=1.0)
    ap.add_argument("--veh-detect-prob", type=float, default=1.0)
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()
    if not sumo_available():
        print("SUMO/traci not available - cannot run the simulation evaluation.", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    model, timing = IntersectionModel.four_way(), SignalTiming()
    noise = SensorNoise(vehicle_detect_prob=args.veh_detect_prob, ped_detect_prob=args.ped_detect_prob)

    agent = None
    if args.checkpoint:
        from src.rl_agent.agent import TrafficSignalAgent

        agent = TrafficSignalAgent.load(args.checkpoint)
        agent.check_compatible(ObservationEncoder(model).dim, model.num_phases)

    all_results = {}
    for sc in args.scenarios:
        def env(interval, sc=sc):
            return lambda: TrafficSignalEnv(SumoBackend(model, sc, noise=noise), timing,
                                            decision_interval_s=interval, episode_s=args.episode_s)
        policies = {"fixed_time": FixedTimePolicy(model), "actuated": ActuatedPolicy(model)}
        factories = {"fixed_time": env(1.0), "actuated": env(1.0)}
        if agent is not None:
            policies["dqn"] = RLPolicy(agent, ObservationEncoder(model))
            factories["dqn"] = env(args.rl_interval)
        all_results[sc] = compare_policies(factories, policies, args.seeds)
        print(f"finished scenario {sc}", flush=True)

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "results": all_results}, fh, indent=1)
    lines = [f"Seeds: {args.seeds}; episode {args.episode_s:.0f}s; mean ± std over seeds.\n"]
    for sc, res in all_results.items():
        lines += [f"\n### {sc}\n", "| controller | " + " | ".join(h for _, h in COLS) + " |",
                  "|---|" + "---|" * len(COLS)]
        for name, m in res.items():
            lines.append(f"| {name} | " + " | ".join(f"{m[k]['mean']:.1f} ± {m[k]['std']:.1f}" for k, _ in COLS) + " |")
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
