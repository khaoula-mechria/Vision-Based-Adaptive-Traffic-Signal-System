#!/usr/bin/env python
"""Train the DQN signal agent.

Examples
    python scripts/train.py --backend queue --episodes 50 --out runs/dqn_queue      # no SUMO needed
    python scripts/train.py --backend sumo  --episodes 150 --out runs/dqn_sumo      # needs SUMO + traci

Training seeds (1000+) and checkpoint-selection seeds (5000+) are disjoint from the final
evaluation seeds used by ``scripts/evaluate.py`` (9000+). Every action still passes through the
``SignalSafetyManager`` during training.
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl_agent import RewardWeights, TrafficSignalEnv  # noqa: E402
from src.rl_agent.agent import DQNConfig, TrafficSignalAgent  # noqa: E402
from src.safety import IntersectionModel, SignalTiming  # noqa: E402
from src.simulation import SCENARIOS, QueueModelBackend, SensorNoise  # noqa: E402

TRAIN_SCENARIOS = ["low", "medium", "high", "rush_hour", "asymmetric", "ped_heavy", "emergency"]
EVAL_SCENARIOS = ["medium", "high", "rush_hour"]


def make_backend(kind: str, model, scenario: str, seed: int, noise: SensorNoise):
    if kind == "sumo":
        from src.simulation import SumoBackend

        return SumoBackend(model, scenario, seed=seed, noise=noise)
    return QueueModelBackend(model, seed=seed, noise=noise)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sumo", "queue"], default="sumo")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--episode-s", type=float, default=1800.0)
    ap.add_argument("--decision-interval", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dqn")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--epsilon-decay-steps", type=int, default=20000)
    ap.add_argument("--ped-detect-prob", type=float, default=1.0, help="train under degraded pedestrian detection")
    ap.add_argument("--veh-detect-prob", type=float, default=1.0)
    ap.add_argument("--ped-weight", type=float, default=1.5, help="reward weight on pedestrian waiting")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)

    model, timing = IntersectionModel.four_way(), SignalTiming()
    noise = SensorNoise(vehicle_detect_prob=args.veh_detect_prob, ped_detect_prob=args.ped_detect_prob)
    train_be = make_backend(args.backend, model, TRAIN_SCENARIOS[0], args.seed, noise)
    eval_be = make_backend(args.backend, model, EVAL_SCENARIOS[0], args.seed, noise)
    weights = RewardWeights(pedestrian_wait=args.ped_weight)
    mk = lambda be: TrafficSignalEnv(be, timing, reward_weights=weights, decision_interval_s=args.decision_interval,
                                      episode_s=args.episode_s, seed=args.seed)
    env, eval_env = mk(train_be), mk(eval_be)

    cfg = DQNConfig(seed=args.seed, epsilon_decay_steps=args.epsilon_decay_steps)
    agent = TrafficSignalAgent(env.observation_dim, env.action_dim, cfg)

    def before_episode(ep: int, seed: int) -> None:
        if args.backend == "sumo":
            train_be.scenario = SCENARIOS[TRAIN_SCENARIOS[ep % len(TRAIN_SCENARIOS)]]

    def eval_setup(i: int, seed: int) -> None:
        if args.backend == "sumo":
            eval_be.scenario = SCENARIOS[EVAL_SCENARIOS[i % len(EVAL_SCENARIOS)]]

    t0 = time.time()

    def log(rec):
        extra = f" eval_return={rec['eval_mean_return']:.1f} eval_wait={rec.get('eval_mean_vehicle_wait_s', 0):.1f}" \
            if "eval_mean_return" in rec else ""
        logging.info("ep %d return %.1f eps %.2f loss %.4f t=%.0fs%s", rec["episode"], rec["train_return"],
                     rec["epsilon"], rec["loss"], time.time() - t0, extra)

    try:
        agent.train(
            env, args.episodes,
            seeds=[1000 + i for i in range(args.episodes)],
            eval_env=eval_env, eval_seeds=[5000, 5001, 5002], eval_every=args.eval_every,
            checkpoint_dir=args.out, log_fn=log, before_episode=before_episode, eval_setup=eval_setup,
        )
    finally:
        env.close()
        eval_env.close()
    with open(os.path.join(args.out, "train_args.json"), "w", encoding="utf-8") as fh:
        json.dump({**vars(args), "dqn": cfg.__dict__, "scenarios": TRAIN_SCENARIOS, "wall_time_s": time.time() - t0},
                  fh, indent=2, default=list)
    logging.info("done; checkpoints in %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
