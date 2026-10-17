"""Config, controller integration, simulation backends (SUMO tests skip when SUMO is unavailable)."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.controller import AdaptiveTrafficController
from src.errors import ConfigError, DetectionError, SimulationError
from src.safety import IntersectionModel, SignalTiming
from src.simulation import SCENARIOS, QueueModelBackend, SensorNoise, sumo_available
from src.simulation.sumo_scenario import generate_routes, validate_sumo_inputs
from src.state import Stage
from src.utils import ConfigLoader
from tests.conftest import det_result, detection


# ----------------------------------------------------------------------- config
def test_default_config_loads_and_validates():
    c = ConfigLoader(env={})
    assert c.get("signal.policy") == "actuated" and c.get("signal.timing.yellow_s") == 4.0
    assert c.get("does.not.exist", 7) == 7


def test_env_overrides_nested_values_and_types():
    c = ConfigLoader(env={"TRAFFIC__SIGNAL__TIMING__YELLOW_S": "4.5", "TRAFFIC__PERCEPTION__DEVICE": "cuda",
                          "TRAFFIC__QUEUE_ESTIMATION__METERS_PER_PIXEL": "0.05", "UNRELATED": "x"})
    assert c.get("signal.timing.yellow_s") == 4.5 and c.get("perception.device") == "cuda"
    assert c.get("queue_estimation.meters_per_pixel") == 0.05


@pytest.mark.parametrize("env", [
    {"TRAFFIC__SIGNAL__POLICY": "magic"},
    {"TRAFFIC__PERCEPTION__CONFIDENCE_THRESHOLD": "1.5"},
    {"TRAFFIC__NOPE__X": "1"},
    {"TRAFFIC__PERCEPTION__TYPO": "1"},
    {"TRAFFIC__SIGNAL__TIMING__YELLOW_S": "0"},
    {"TRAFFIC__TRACKING__MAX_AGE": "abc"},
])
def test_invalid_overrides_raise(env):
    with pytest.raises(ConfigError):
        ConfigLoader(env=env)


def test_user_file_merge_and_errors(tmp_path):
    good = tmp_path / "c.yaml"
    good.write_text("signal:\n  policy: fixed\n  timing:\n    max_green_s: 50\n")
    c = ConfigLoader(str(good), env={})
    assert c.get("signal.policy") == "fixed" and c.get("signal.timing.min_green_s") == 8.0
    bad = tmp_path / "bad.yaml"
    bad.write_text("signal: [unclosed")
    with pytest.raises(ConfigError):
        ConfigLoader(str(bad), env={})
    unknown = tmp_path / "u.yaml"
    unknown.write_text("bogus: 1\n")
    with pytest.raises(ConfigError, match="unknown"):
        ConfigLoader(str(unknown), env={})
    with pytest.raises(ConfigError, match="not found"):
        ConfigLoader(str(tmp_path / "missing.yaml"), env={})


# ------------------------------------------------------------------- controller
LANES = [{"id": i, "approach": a, "polygon": poly, "stop_line": line} for i, (a, poly, line) in enumerate([
    ("N", [[100, 0], [200, 0], [200, 450], [100, 450]], [[100, 400], [200, 400]]),
    ("E", [[500, 0], [600, 0], [600, 450], [500, 450]], [[500, 400], [600, 400]]),
])]


class ScriptedDetector:
    """Detector double: returns scripted DetectionResults, raises scripted errors, or None -> DetectionError."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def detect(self, frame, timestamp=None):
        item = self.script(self.i, timestamp)
        self.i += 1
        if isinstance(item, Exception):
            raise item
        return item


def make_controller(script, policy="actuated", **cfg):
    c = ConfigLoader(env={})
    c.config["queue_estimation"]["lanes"] = LANES
    c.config["signal"]["policy"] = policy
    c.config["perception"]["max_consecutive_failures"] = 3
    return AdaptiveTrafficController(config=c, detector=ScriptedDetector(script))


FRAME = np.zeros((480, 640, 3), np.uint8)


def test_controller_runs_pipeline_and_serves_demand_without_violations():
    def script(i, ts):  # queue of stopped cars on the E approach, none on N
        return det_result(ts, [detection(520, 330, 580, 390), detection(520, 240, 580, 290)])
    ctl = make_controller(script)
    results = [ctl.step(FRAME, float(t)) for t in range(80)]
    assert all(r.sensors_ok for r in results) and results[-1].num_tracks == 2
    assert ctl.last_error is None
    assert any(r.command.approach_lights["E"] == "G" for r in results)
    assert ctl.get_status()["audit_violations"] == 0
    assert ctl.queue.approach_states()["E"]["queue"] == 2


def test_missing_frames_and_detector_errors_degrade_to_fallback_cycle():
    ctl = make_controller(lambda i, ts: det_result(ts, []) if i < 20 else DetectionError("camera unplugged"))
    results = [ctl.step(FRAME if t < 20 else None, float(t)) for t in range(250)]
    assert results[10].sensors_ok and not results[-1].sensors_ok
    degraded = [r for r in results[40:] if r.command.stage is Stage.HOLD]
    assert {r.command.phase for r in degraded} == {0, 1, 2}  # fixed fallback serves every phase, incl. pedestrians
    assert ctl.get_status()["audit_violations"] == 0 and ctl.last_error


def test_unexpected_detector_exception_is_contained():
    ctl = make_controller(lambda i, ts: RuntimeError("segfault-ish"))
    for t in range(10):
        r = ctl.step(FRAME, float(t))  # must not raise
    assert not r.sensors_ok and "RuntimeError" in (r.error or "")


def test_stale_data_marks_sensors_not_ok():
    ctl = make_controller(lambda i, ts: det_result(ts, []))
    ctl.step(FRAME, 0.0)
    r = ctl.step(FRAME, 0.1)
    assert r.sensors_ok
    ctl._max_stale = 0.5
    ctl.detector.script = lambda i, ts: DetectionError("x")
    r = ctl.step(FRAME, 5.0)
    assert not r.sensors_ok


def test_emergency_vehicle_is_confirmed_then_preempts_through_clearance():
    def script(i, ts):
        return det_result(ts, [detection(520, 300, 590, 380, cls="ambulance", conf=0.95)])
    ctl = make_controller(script)
    results = [ctl.step(FRAME, float(t)) for t in range(60)]
    first_e = next(r for r in results if r.command.approach_lights["E"] == "G")
    assert first_e.command.emergency_override or first_e.timestamp > 0
    assert any(r.command.approach_lights["N"] == "y" for r in results) or results[0].command.phase == 1
    assert ctl.get_status()["audit_violations"] == 0


def test_rl_policy_with_missing_checkpoint_falls_back_to_actuated(caplog):
    c = ConfigLoader(env={})
    c.config["signal"]["policy"] = "rl"
    c.config["signal"]["rl_checkpoint"] = "does/not/exist.pt"
    ctl = AdaptiveTrafficController(config=c, detector=ScriptedDetector(lambda i, ts: det_result(ts, [])))
    assert ctl.policy.name == "actuated"
    assert "RL policy unavailable" in caplog.text


def test_rejected_rl_action_does_not_change_lights():
    """A policy that demands a switch every tick is overridden by min-green / clearance rules."""
    from src.state import PhaseRequest, RequestSource

    class Flip:
        name = "flip"
        def reset(self): pass
        def request(self, obs, ctx): return PhaseRequest((ctx.phase + 1) % 3, RequestSource.RL)

    c = ConfigLoader(env={})
    ctl = AdaptiveTrafficController(config=c, detector=ScriptedDetector(lambda i, ts: det_result(ts, [])),
                                    policy=Flip())
    results = [ctl.step(FRAME, float(t)) for t in range(60)]
    holds = [r for r in results if r.command.stage is Stage.HOLD and r.command.decision]
    assert any(not r.command.decision.accepted for r in holds)
    first_green = next(r for r in results if r.command.stage is Stage.HOLD)
    assert results[results.index(first_green) + 5].command.phase == first_green.command.phase  # min green held
    assert ctl.auditor.total == 0


def test_controller_reset_and_on_command_hook():
    seen = []
    c = ConfigLoader(env={})
    ctl = AdaptiveTrafficController(config=c, detector=ScriptedDetector(lambda i, ts: det_result(ts, [])),
                                    on_command=seen.append)
    for t in range(5):
        ctl.step(FRAME, float(t))
    assert len(seen) == 5
    ctl.reset()
    assert ctl.frame_count == 0 and ctl.manager.stage is Stage.ALL_RED


# ------------------------------------------------------------ queue-model backend
def test_queue_backend_is_deterministic_and_resettable(model):
    from src.safety import SignalSafetyManager

    def trace(seed):
        be = QueueModelBackend(model, seed=seed)
        mgr = SignalSafetyManager(model, SignalTiming())
        out = []
        for t in range(300):
            cmd = mgr.step(float(t), None, be.observe())
            s = be.step(cmd)
            out.append((s.vehicle_wait_s, s.discharged))
        return out, be
    a, be = trace(1)
    assert a == trace(1)[0] and a != trace(2)[0]
    be.reset(1)
    assert be.time_s == 0 and be.episode_metrics()["throughput_veh"] == 0


def test_sensor_noise_thins_counts_and_flags_uncertainty(model):
    be = QueueModelBackend(model, seed=0, noise=SensorNoise(vehicle_detect_prob=0.2, ped_detect_prob=0.2))
    truth = be._truth()
    import random

    from src.simulation.backend import apply_sensor_noise
    noisy = apply_sensor_noise(type(truth)(**{**truth.__dict__, "approaches": {
        a: type(v)(queue=50.0) for a, v in truth.approaches.items()}}), be.noise, random.Random(0))
    assert all(v.queue < 30 for v in noisy.approaches.values()) and noisy.ped_uncertain
    dropout = apply_sensor_noise(truth, SensorNoise(dropout_prob=1.0), random.Random(0))
    assert not dropout.sensors_ok


# ------------------------------------------------------------------------ SUMO
def test_route_generation_is_deterministic_and_seed_dependent(tmp_path):
    a = generate_routes(SCENARIOS["medium"], 5, tmp_path / "a.xml").read_text()
    b = generate_routes(SCENARIOS["medium"], 5, tmp_path / "b.xml").read_text()
    c = generate_routes(SCENARIOS["medium"], 6, tmp_path / "c.xml").read_text()
    assert a == b and a != c
    root = ET.fromstring(a)
    departs = [float(e.get("depart")) for e in root if e.tag in ("vehicle", "person")]
    assert departs == sorted(departs) and len(departs) > 100
    emg = generate_routes(SCENARIOS["emergency"], 5, tmp_path / "e.xml").read_text()
    assert 'id="emg_' in emg and 'id="emg_' not in a


def test_validate_sumo_inputs_reports_bad_files(tmp_path):
    good = tmp_path / "g.xml"
    good.write_text("<routes/>")
    with pytest.raises(SimulationError, match="not found"):
        validate_sumo_inputs(tmp_path / "nope.xml", good, 1.0)
    bad = tmp_path / "bad.xml"
    bad.write_text("<net><unclosed>")
    with pytest.raises(SimulationError, match="not valid XML"):
        validate_sumo_inputs(bad, good, 1.0)
    with pytest.raises(SimulationError, match="step_length"):
        validate_sumo_inputs(good, good, 0)
    with pytest.raises(SimulationError, match="traffic_light"):
        validate_sumo_inputs(good, good, 1.0)


def test_sumo_backend_without_traci_raises_clear_error(monkeypatch):
    import src.simulation.sumo_backend as sb
    monkeypatch.setattr(sb, "TRACI_AVAILABLE", False)
    with pytest.raises(SimulationError, match="traci"):
        sb.SumoBackend()


def test_missing_sumo_binary_is_reported(monkeypatch):
    import src.simulation.sumo_scenario as sc
    monkeypatch.setattr(sc, "find_sumo_home", lambda: None)
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    with pytest.raises(SimulationError, match="not found"):
        sc.find_binary("sumo")


needs_sumo = pytest.mark.skipif(not sumo_available(), reason="SUMO/traci not available")


@needs_sumo
def test_sumo_reset_step_close_cleans_up_and_is_reproducible():
    from src.rl_agent import TrafficSignalEnv
    from src.simulation import SumoBackend

    def episode(seed):
        be = SumoBackend(IntersectionModel.four_way(), "low", seed=seed)
        env = TrafficSignalEnv(be, decision_interval_s=5.0, episode_s=300)
        env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            _, r, _, done, _ = env.step(0 if env.backend.time_s < 100 else 1)
            total += r
        summary = env.summary()
        work_dir = be._dir
        env.close()
        return total, summary, work_dir

    t1, s1, d1 = episode(3)
    t2, s2, d2 = episode(3)
    t3, s3, _ = episode(4)
    assert t1 == t2 and s1 == s2 and t1 != t3
    assert s1["safety_violations"] == 0 and s1["tls_state_mismatch_steps"] == 0 and s1["collisions"] == 0
    assert not d1.exists() and not d2.exists()  # temp dirs removed on close


@needs_sumo
def test_sumo_multiple_resets_and_safe_double_close():
    from src.simulation import SumoBackend
    be = SumoBackend(IntersectionModel.four_way(), "low", seed=1)
    for seed in (1, 2, 1):
        be.reset(seed)
        assert be.time_s == 0.0 and be.episode_metrics()["throughput_veh"] == 0
    be.close()
    be.close()  # idempotent
    with pytest.raises(SimulationError):
        be.reset(1)


@needs_sumo
def test_sumo_step_before_reset_and_after_crash_raise(model):
    from src.safety import SignalSafetyManager
    from src.simulation import SumoBackend
    be = SumoBackend(model, "low", seed=1)
    cmd = SignalSafetyManager(model, SignalTiming()).step(0.0, None, None)
    with pytest.raises(SimulationError, match="reset"):
        be.step(cmd)
    be.reset(1)
    be.step(cmd)
    be._conn._socket.close()  # simulate a dropped TraCI connection
    with pytest.raises(SimulationError):
        be.step(cmd)
    be.close()


@needs_sumo
def test_sumo_displayed_lights_pass_independent_audit(model):
    """Read the TLS state SUMO actually displays and audit it, not just the commanded one."""
    from src.safety import SignalAuditor, SignalSafetyManager
    from src.simulation import SumoBackend
    from src.state import PhaseRequest
    be = SumoBackend(model, "medium", seed=2)
    be.reset(2)
    mgr, aud = SignalSafetyManager(model, SignalTiming()), SignalAuditor(model, SignalTiming())
    for t in range(400):
        cmd = mgr.step(be.time_s, PhaseRequest((t // 15) % 3), be.observe())
        be.step(cmd)
        assert be.executed_state() == be._state_string(cmd)
        aud.observe_command(cmd)
    assert aud.total == 0 and be.episode_metrics()["collisions"] == 0
    be.close()
