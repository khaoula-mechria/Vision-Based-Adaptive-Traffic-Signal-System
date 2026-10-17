"""RL: observation, reward, env, agent, policies, and the controller's behaviour when RL is rejected."""

import numpy as np
import pytest

from src.control import ActuatedPolicy, FixedTimePolicy, GuardedPolicy
from src.errors import ConfigError, ModelNotFoundError
from src.evaluation import run_policy_episode
from src.rl_agent import ObservationEncoder, RewardCalculator, TrafficSignalEnv
from src.rl_agent.reward import IntervalStats, RewardWeights
from src.safety import IntersectionModel, SignalSafetyManager, SignalTiming
from src.simulation import QueueModelBackend, SensorNoise
from src.state import PhaseRequest, RequestSource
from tests.conftest import make_obs

torch = pytest.importorskip("torch")
from src.rl_agent.agent import DQNConfig, TrafficSignalAgent  # noqa: E402


def make_env(seed=0, episode_s=200.0, interval=5.0, **kw):
    return TrafficSignalEnv(QueueModelBackend(IntersectionModel.four_way(), seed=seed, **kw),
                            decision_interval_s=interval, episode_s=episode_s, seed=seed)


# ------------------------------------------------------------------ observation
def test_observation_dimension_derived_from_model(model):
    enc = ObservationEncoder(model)
    assert enc.dim == 3 * 4 + 2 * 4 + 3 + 2
    ctx = SignalSafetyManager(model, SignalTiming()).context(0.0)
    v = enc.encode(make_obs(model, 0.0, queues={"N": 5}), ctx)
    assert v.shape == (enc.dim,) and v.dtype == np.float32 and 0 <= v.min() and v.max() <= 1


def test_observation_clipped_and_nan_safe(model):
    enc = ObservationEncoder(model)
    ctx = SignalSafetyManager(model, SignalTiming()).context(0.0)
    obs = make_obs(model, 0.0, queues={"N": 10_000}, ped_wait=float("nan"))
    v = enc.encode(obs, ctx)
    assert np.all(np.isfinite(v)) and v.max() <= 1.0


def test_encoder_adapts_to_other_geometries():
    m = IntersectionModel.from_dict({
        "approaches": ["A", "B", "C"], "crosswalks": [],
        "phases": [{"name": "A", "green": ["A"]}, {"name": "BC", "green": ["B", "C"]}],
        "conflicts": [["A", "B"], ["A", "C"]]})
    assert ObservationEncoder(m).dim == 9 + 0 + 2 + 2


# ----------------------------------------------------------------------- reward
def stats(**kw):
    base = dict(vehicle_wait_s=0, pedestrian_wait_s=0, mean_queue=0, discharged=0, max_approach_wait_s=0,
                phase_changes=0, rejected=False, violations=0, emergency_wait_s=0, duration_s=5.0)
    base.update(kw)
    return IntervalStats(**base)


def test_reward_signs_and_monotonicity():
    r = RewardCalculator()
    assert r(stats()) == 0.0
    assert r(stats(vehicle_wait_s=50)) < r(stats(vehicle_wait_s=10)) < 0
    assert r(stats(pedestrian_wait_s=20)) < 0
    assert r(stats(discharged=5)) > 0
    assert r(stats(phase_changes=1)) < 0
    assert r(stats(max_approach_wait_s=100)) < r(stats(max_approach_wait_s=10))
    assert r(stats(violations=1)) < r(stats(rejected=True)) < 0


def test_reward_is_stationary_and_clipped():
    r = RewardCalculator(RewardWeights(clip=5.0))
    a = [r(stats(vehicle_wait_s=30)) for _ in range(5)]
    assert len(set(a)) == 1  # no running normalisation state
    assert r(stats(vehicle_wait_s=1e9)) == -10.0
    assert set(r.breakdown(stats(vehicle_wait_s=10))) >= {"vehicle_wait", "throughput", "fairness", "switching"}


# -------------------------------------------------------------------------- env
def test_env_spaces_reset_step_and_truncation():
    env = make_env(episode_s=60)
    obs, info = env.reset(seed=3)
    assert obs.shape == (env.observation_dim,) and env.observation_space.contains(obs)
    assert env.action_space.n == 3
    done, steps = False, 0
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)
        done = term or trunc
        steps += 1
    assert trunc and not term and steps == 12 and "episode" in info
    with pytest.raises(RuntimeError):
        env.step(0)


def test_env_is_deterministic_for_a_seed():
    def roll(seed):
        env = make_env(seed=seed, episode_s=120)
        env.reset(seed=seed)
        rewards = [env.step(i % 3)[1] for i in range(24)]
        return rewards, env.summary()
    assert roll(5) == roll(5)
    assert roll(5) != roll(6)


def test_env_reset_clears_state():
    env = make_env(episode_s=100)
    env.reset(seed=1)
    for _ in range(10):
        env.step(1)
    env.reset(seed=1)
    assert env.backend.time_s == 0 and env.manager.stats["phase_changes"] == 0 and env.auditor.total == 0


def test_adversarial_actions_never_violate_safety_and_are_overridden():
    """The RL action is only a request: flipping phases every step must not break any invariant."""
    env = make_env(seed=2, episode_s=900, interval=1.0)
    env.reset(seed=2)
    done, rejected = False, 0
    while not done:
        action = (int(env.backend.time_s) % 3)  # constantly changing, often too early
        _, _, _, done, info = env.step(action)
        rejected += info["request_rejected"]
    s = env.summary()
    assert s["safety_violations"] == 0 and s["ped_conflict_steps"] == 0
    assert rejected > 0 and s["requests_overridden"] > 0


def test_invalid_action_values_are_handled_by_safety_layer():
    env = make_env(episode_s=60)
    env.reset(seed=0)
    for bad in (-5, 7, 3):
        _, r, _, _, info = env.step(bad)
        assert info["request_rejected"] or info["stage"] != "hold"
    assert env.summary()["safety_violations"] == 0


def test_env_under_sensor_dropout_stays_safe():
    env = make_env(seed=4, episode_s=600, noise=SensorNoise(0.7, 0.5, 0.5, dropout_prob=0.3))
    env.reset(seed=4)
    done = False
    while not done:
        _, _, _, done, _ = env.step(0)
    assert env.summary()["safety_violations"] == 0


# ------------------------------------------------------------------------ policies
def test_baselines_run_and_serve_all_phases():
    for mk in (lambda m: FixedTimePolicy(m), lambda m: ActuatedPolicy(m)):
        env = make_env(seed=1, episode_s=900, interval=1.0)
        res = run_policy_episode(env, mk(env.model), seed=1)
        assert res["safety_violations"] == 0 and res["throughput_veh"] > 0 and res["phase_switches"] >= 3


def test_actuated_prefers_phase_with_demand(model):
    mgr = SignalSafetyManager(model, SignalTiming())
    for t in range(20):
        mgr.step(float(t), None, make_obs(model, float(t)))
    obs = make_obs(model, 20.0, queues={"E": 8, "W": 6})
    req = ActuatedPolicy(model, gap_s=0).request(obs, mgr.context(20.0))
    assert req.phase == 1 and req.source is RequestSource.ACTUATED


def test_guarded_policy_falls_back_on_exception_bad_output_and_bad_sensors(model):
    class Boom:
        name = "boom"
        def reset(self): pass
        def request(self, obs, ctx): raise RuntimeError("model exploded")

    class Garbage(Boom):
        def request(self, obs, ctx): return PhaseRequest(phase=None)  # type: ignore[arg-type]

    ctx = SignalSafetyManager(model, SignalTiming()).context(0.0)
    obs = make_obs(model, 0.0)
    for primary in (Boom(), Garbage()):
        g = GuardedPolicy(primary, FixedTimePolicy(model))
        assert g.request(obs, ctx).source is RequestSource.FALLBACK and g.fallback_count == 1
    ok = GuardedPolicy(ActuatedPolicy(model), FixedTimePolicy(model))
    assert ok.request(make_obs(model, 0.0, sensors_ok=False), ctx).source is RequestSource.FALLBACK


# ------------------------------------------------------------------------ agent
def small_agent(**kw):
    cfg = DQNConfig(learn_start=64, batch_size=32, buffer_size=1000, hidden_dims=(16, 16),
                    epsilon_decay_steps=100, target_update_steps=10, **kw)
    return TrafficSignalAgent(28 - 3, 3, cfg)


def test_agent_action_shapes_and_validation():
    a = small_agent()
    s = np.random.default_rng(0).random(25).astype(np.float32)
    assert a.select_action(s) in (0, 1, 2)
    with pytest.raises(ValueError):
        a.select_action(np.zeros(5, np.float32))


def test_greedy_action_is_deterministic_and_eval_mode_never_explores():
    a = small_agent()
    s = np.random.default_rng(1).random(25).astype(np.float32)
    assert len({a.select_action(s, training=False) for _ in range(50)}) == 1
    assert len({a.select_action(s, training=True) for _ in range(300)}) > 1  # epsilon=1 explores


def test_epsilon_schedule_is_linear_and_bounded():
    a = small_agent()
    assert a.epsilon == 1.0
    a.env_steps = 50
    assert a.epsilon == pytest.approx(0.525)
    a.env_steps = 10_000
    assert a.epsilon == pytest.approx(a.cfg.epsilon_end)


def test_truncation_is_not_terminal_in_replay():
    a = small_agent()
    env = make_env(episode_s=20)
    s, _ = env.reset(seed=0)
    done = False
    while not done:
        s2, r, term, trunc, _ = env.step(0)
        a.observe(s, 0, r, s2, term)
        s, done = s2, term or trunc
    assert a.replay.d[: len(a.replay)].sum() == 0


def test_training_learns_something_and_updates_target_net():
    a = small_agent()
    env = make_env(episode_s=300)
    before = [p.clone() for p in a.q_net.parameters()]
    hist = a.train(env, episodes=3, seeds=[1, 2, 3])
    assert len(hist) == 3 and a.grad_steps > 0 and np.isfinite(hist[-1]["loss"])
    assert any(not torch.equal(b, p) for b, p in zip(before, a.q_net.parameters()))
    assert any(not torch.equal(t, q) for t, q in zip(a.target_net.parameters(), a.q_net.parameters())) or a.grad_steps % 10 == 0


def test_evaluation_does_not_train_or_store_experience():
    a = small_agent()
    env = make_env(episode_s=100)
    steps, size, grads = a.env_steps, len(a.replay), a.grad_steps
    weights = [p.clone() for p in a.q_net.parameters()]
    res = a.evaluate(env, [1, 2])
    assert (a.env_steps, len(a.replay), a.grad_steps) == (steps, size, grads)
    assert all(torch.equal(w, p) for w, p in zip(weights, a.q_net.parameters()))
    assert "mean_return" in res and "safety_violations" in res and res["safety_violations"] == 0


def test_training_is_reproducible_with_the_same_seed():
    def run(seed):
        a = TrafficSignalAgent(25, 3, DQNConfig(seed=seed, learn_start=64, batch_size=32, hidden_dims=(16,)))
        h = a.train(make_env(seed=seed, episode_s=200), episodes=2, seeds=[7, 8])
        return [r["train_return"] for r in h], [p.detach().numpy().copy() for p in a.q_net.parameters()]
    (r1, w1), (r2, w2) = run(11), run(11)
    assert r1 == r2 and all(np.array_equal(x, y) for x, y in zip(w1, w2))


def test_checkpoint_roundtrip_and_failures(tmp_path):
    a = small_agent()
    a.env_steps = 123
    path = str(tmp_path / "ck" / "m.pt")
    a.save(path)
    b = TrafficSignalAgent.load(path)
    s = np.random.default_rng(3).random(25).astype(np.float32)
    assert b.env_steps == 123 and b.select_action(s) == a.select_action(s)
    with pytest.raises(ModelNotFoundError):
        TrafficSignalAgent.load(str(tmp_path / "missing.pt"))
    with pytest.raises(ConfigError):
        b.check_compatible(40, 3)


def test_dqn_config_validation():
    with pytest.raises(ConfigError):
        TrafficSignalAgent(5, 3, DQNConfig(gamma=1.5))
    with pytest.raises(ConfigError):
        TrafficSignalAgent(5, 3, DQNConfig(epsilon_start=0.1, epsilon_end=0.5))
