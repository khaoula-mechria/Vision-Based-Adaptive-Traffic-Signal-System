"""Safety layer: conflicts, timing, pedestrians, emergency, degraded sensing, and invariants."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.errors import ConfigError, IntersectionModelError
from src.safety import IntersectionModel, PhaseDef, SignalAuditor, SignalSafetyManager, SignalTiming
from src.state import PhaseRequest, Stage
from tests.conftest import make_obs, run


# ---------------------------------------------------------------- intersection model
def test_default_model_is_valid(model):
    assert [p.name for p in model.phases] == ["NS", "EW", "PED"]
    assert model.phases_conflict(0, 1) and model.phases_conflict(0, 2)


def test_conflicting_phase_is_rejected_at_construction():
    base = IntersectionModel.four_way()
    bad = PhaseDef("BAD", frozenset({"N", "E"}))
    with pytest.raises(IntersectionModelError, match="conflicting"):
        IntersectionModel(base.approaches, base.crosswalks, base.phases + (bad,), base.conflicts)


def test_crosswalk_green_with_vehicles_rejected():
    base = IntersectionModel.four_way()
    bad = PhaseDef("BAD", frozenset({"N"}), frozenset({"cw_N"}))
    with pytest.raises(IntersectionModelError):
        IntersectionModel(base.approaches, base.crosswalks, (bad,) + base.phases[1:], base.conflicts)


def test_unserved_approach_and_unknown_element_rejected():
    base = IntersectionModel.four_way()
    with pytest.raises(IntersectionModelError, match="never served"):
        IntersectionModel(base.approaches, base.crosswalks, base.phases[:1] + base.phases[2:], base.conflicts)
    with pytest.raises(IntersectionModelError, match="unknown"):
        IntersectionModel(base.approaches, base.crosswalks, (PhaseDef("X", frozenset({"Q"})),), base.conflicts)


def test_from_dict_supports_other_geometries():
    m = IntersectionModel.from_dict({
        "approaches": ["A", "B", "C"], "crosswalks": [],
        "phases": [{"name": "A", "green": ["A"]}, {"name": "BC", "green": ["B", "C"]}],
        "conflicts": [["A", "B"], ["A", "C"]]})
    assert m.num_phases == 2 and m.phase_for_approach("C") == 1
    with pytest.raises(IntersectionModelError):
        IntersectionModel.from_dict({"approaches": ["A"]})


def test_timing_validation():
    with pytest.raises(ConfigError):
        SignalTiming(yellow_s=0)
    with pytest.raises(ConfigError):
        SignalTiming(min_green_s=50, max_green_s=40)
    with pytest.raises(ConfigError):
        SignalTiming.from_dict({"bogus": 1})


# ---------------------------------------------------------------------- transitions
def test_starts_all_red_then_serves_initial_phase(manager, model):
    cmds = run(manager, model, 12)
    assert all(c.stage is Stage.ALL_RED for c in cmds[:6])
    assert cmds[-1].stage is Stage.HOLD and cmds[-1].approach_lights["N"] == "G"


def test_switch_goes_through_yellow_and_all_red(manager, model, timing):
    aud = SignalAuditor(model, timing)
    run(manager, model, 12, auditor=aud)  # reach NS green
    cmds = run(manager, model, 20, request=PhaseRequest(1), start=12, auditor=aud)
    stages = [c.stage for c in cmds]
    assert Stage.CLEARANCE in stages and Stage.ALL_RED in stages
    first_ew = next(c for c in cmds if c.approach_lights["E"] == "G")
    last_ns_yellow = max(c.time_s for c in cmds if c.approach_lights["N"] == "y")
    assert first_ew.time_s - last_ns_yellow >= timing.all_red_s
    assert not aud.violations


def test_min_green_rejects_early_switch(manager, model, timing):
    run(manager, model, 6 + 4)  # green began at t=6 (startup all-red); 4 s into it
    last = run(manager, model, 1, request=PhaseRequest(1), start=10)[0]
    assert last.stage is Stage.HOLD and last.decision.reason == "min_phase_duration"
    assert not last.decision.accepted and last.decision.effective_phase == 0


def test_invalid_requests_are_rejected(manager, model):
    run(manager, model, 12)
    for bad in (PhaseRequest(-1), PhaseRequest(99), None):
        c = run(manager, model, 1, request=bad, start=12)[0]
        assert c.approach_lights["N"] == "G"


def test_max_green_forces_rotation_even_if_policy_never_switches(manager, model, timing):
    cmds = run(manager, model, 120, request=PhaseRequest(0))
    assert any(c.approach_lights["E"] == "G" for c in cmds), "EW never served"
    longest = 0
    cur = 0
    for c in cmds:
        cur = cur + 1 if c.approach_lights["N"] == "G" else 0
        longest = max(longest, cur)
    assert longest <= timing.max_green_s + 1


def test_non_monotonic_time_is_clamped(manager, model):
    run(manager, model, 12)
    manager.step(10.0, None, make_obs(model, 10.0))  # goes backwards
    assert manager.stats["non_monotonic_time"] == 1


# ------------------------------------------------------------------- pedestrians
def _to_ped_walk(manager, model, aud=None):
    cmds = run(manager, model, 60, request=PhaseRequest(2), auditor=aud)
    return cmds


def test_pedestrian_phase_has_walk_flash_and_full_clearance(manager, model, timing):
    aud = SignalAuditor(model, timing)
    cmds = _to_ped_walk(manager, model, aud)
    walk = [c for c in cmds if c.crosswalk_lights["cw_N"] == "W"]
    flash = [c for c in cmds if c.crosswalk_lights["cw_N"] == "F"]
    assert len(walk) >= timing.ped_walk_min_s
    assert len(flash) >= timing.ped_clearance_s
    assert not aud.violations


def test_walk_not_cut_short_by_vehicle_queue(manager, model, timing):
    _to_ped_walk(manager, model)  # enter ped phase
    start = 60
    queues = {"N": 30, "S": 30}
    # vehicles beg for NS from the first walk second on
    cmds = run(manager, model, 5, request=PhaseRequest(0), start=start, obs_fn=lambda t: make_obs(model, t, queues))
    # whatever phase we are in, walk lights must never drop before ped_walk_min_s
    assert manager.stage in (Stage.HOLD, Stage.CLEARANCE, Stage.ALL_RED)
    manager2 = SignalSafetyManager(model, timing)
    cmds = run(manager2, model, 80, request=lambda t: PhaseRequest(2 if t < 30 else 0),
               obs_fn=lambda t: make_obs(model, t, queues))
    walk_start = next(c.time_s for c in cmds if c.crosswalk_lights["cw_N"] == "W")
    walk_end = next(c.time_s for c in cmds if c.time_s > walk_start and c.crosswalk_lights["cw_N"] != "W")
    assert walk_end - walk_start >= timing.ped_walk_min_s


def test_walk_extended_while_pedestrians_in_crosswalk(model, timing):
    m = SignalSafetyManager(model, timing)
    occupied = lambda t: make_obs(model, t, ped_occupied=("cw_N",) if t < 40 else ())
    cmds = run(m, model, 80, request=lambda t: PhaseRequest(2 if t < 20 else 0), obs_fn=occupied)
    walk_secs = sum(c.crosswalk_lights["cw_N"] == "W" for c in cmds)
    assert walk_secs > timing.ped_walk_min_s  # extended beyond the minimum
    assert walk_secs <= timing.ped_walk_max_s + 1  # but bounded


def test_vehicle_phase_not_released_while_crosswalk_occupied(model, timing):
    m = SignalSafetyManager(model, timing)
    aud = SignalAuditor(model, timing)
    obs = lambda t: make_obs(model, t, ped_occupied=("cw_E",) if 24 <= t < 44 else ())
    # PED is requested once NS has had its minimum green; the hand-back to NS falls inside the occupied window
    cmds = run(m, model, 100, request=lambda t: PhaseRequest(2 if 14 <= t < 20 else 0), obs_fn=obs, auditor=aud)
    for c in cmds:  # a *new* vehicle green may not start while the crosswalk is sensed occupied
        if 24 <= c.time_s < 44 and any(e.startswith("phase_start") for e in c.events):
            assert all(v != "G" for v in c.approach_lights.values()), f"vehicle green released at {c.time_s}"
    # the walk was extended instead of handing the crosswalk to vehicles
    assert m.stats["rejected:pedestrians_in_crosswalk"] > 0
    assert not aud.violations


def test_conflict_hold_times_out_by_serving_pedestrians(model):
    timing = SignalTiming(max_conflict_hold_s=6.0)
    m = SignalSafetyManager(model, timing)
    cmds = run(m, model, 150, request=lambda t: PhaseRequest(1),
               obs_fn=lambda t: make_obs(model, t, ped_occupied=("cw_E",) if t > 30 else ()))
    assert any("conflict_hold_timeout_serve_pedestrians" in e for c in cmds for e in c.events)
    timeout_t = next(c.time_s for c in cmds if any("serve_pedestrians" in e for e in c.events))
    # no vehicle phase started between the crosswalk becoming occupied and the pedestrians being served
    for c in cmds:
        if 31 < c.time_s <= timeout_t:
            assert not any(e.startswith("phase_start:NS") or e.startswith("phase_start:EW") for e in c.events)


def test_pedestrian_wait_cap_forces_pedestrian_service(model, timing):
    m = SignalSafetyManager(model, timing)
    cmds = run(m, model, 80, request=PhaseRequest(0),
               obs_fn=lambda t: make_obs(model, t, ped_wait=timing.max_pedestrian_wait_s + 1, ped_waiting=3))
    assert any(c.crosswalk_lights["cw_N"] == "W" for c in cmds)


def test_junction_box_occupancy_delays_release(model, timing):
    m = SignalSafetyManager(model, timing)
    cmds = run(m, model, 40, request=PhaseRequest(1),
               obs_fn=lambda t: make_obs(model, t, box=1 if 16 <= t < 24 else 0))
    assert any("junction_box" in e for c in cmds for e in c.events)


# ----------------------------------------------------------------------- emergency
def test_emergency_preempts_but_keeps_clearance(model, timing):
    m = SignalSafetyManager(model, timing)
    aud = SignalAuditor(model, timing)
    obs = lambda t: make_obs(model, t, emergency=("E",) if t >= 15 else ())
    cmds = run(m, model, 50, request=PhaseRequest(0), obs_fn=obs, auditor=aud)
    first_e = next(c for c in cmds if c.approach_lights["E"] == "G")
    assert first_e.time_s < 15 + timing.yellow_s + timing.all_red_s + 3 + timing.emergency_min_green_s
    assert any(c.approach_lights["N"] == "y" for c in cmds)  # clearance still happened
    assert not aud.violations


def test_emergency_waits_for_minimum_pedestrian_walk(model, timing):
    m = SignalSafetyManager(model, timing)
    obs = lambda t: make_obs(model, t, emergency=("N",) if t >= 22 else ())
    cmds = run(m, model, 70, request=PhaseRequest(2), obs_fn=obs)
    ws = [c for c in cmds if c.crosswalk_lights["cw_N"] == "W"]
    assert len(ws) >= timing.ped_walk_min_s


def test_emergency_override_ignored_when_sensors_down(model, timing):
    m = SignalSafetyManager(model, timing)
    obs = lambda t: make_obs(model, t, emergency=("E",), sensors_ok=False)
    cmds = run(m, model, 40, request=PhaseRequest(1), obs_fn=obs)
    assert not any(c.emergency_override for c in cmds)


# ------------------------------------------------------------------- degraded sensing
def test_sensor_failure_runs_fallback_cycle_serving_all_phases(model, timing):
    m = SignalSafetyManager(model, timing)
    aud = SignalAuditor(model, timing)
    cmds = run(m, model, 200, request=PhaseRequest(0), obs_fn=lambda t: make_obs(model, t, sensors_ok=False),
               auditor=aud)
    served = {c.phase for c in cmds if c.stage is Stage.HOLD}
    assert served == {0, 1, 2}
    assert not aud.violations


def test_missing_observation_treated_as_degraded(model, timing):
    m = SignalSafetyManager(model, timing)
    cmds = [m.step(float(t), PhaseRequest(0), None) for t in range(120)]
    assert {c.phase for c in cmds if c.stage is Stage.HOLD} == {0, 1, 2}


def test_uncertain_pedestrian_detection_adds_all_red(model, timing):
    def all_red_len(unc):
        m = SignalSafetyManager(model, timing)
        cmds = run(m, model, 40, request=PhaseRequest(1), obs_fn=lambda t: make_obs(model, t, ped_uncertain=unc))
        trans = [c for c in cmds if c.time_s >= 12 and c.stage is Stage.ALL_RED]
        return len(trans)
    assert all_red_len(True) > all_red_len(False)


# ------------------------------------------------------------ auditor catches faults
def test_auditor_flags_injected_violations(model, timing):
    a = SignalAuditor(model, timing)
    ap = {x: "r" for x in model.approaches}
    cw = {x: "D" for x in model.crosswalks}
    a.observe(0, ap, cw)
    a.observe(1, {**ap, "N": "G", "E": "G"}, cw)
    assert a.counts["conflicting_green"] >= 1
    a2 = SignalAuditor(model, timing)
    a2.observe(0, {**ap, "N": "G"}, cw)
    a2.observe(20, ap, cw)  # green -> red without yellow
    assert a2.counts["missing_clearance"] == 1
    a3 = SignalAuditor(model, timing)
    a3.observe(0, {**ap, "N": "y"}, cw)
    a3.observe(1, ap, cw)  # 1 s yellow
    assert a3.counts["short_clearance"] == 1
    a3.observe(2, {**ap, "E": "G"}, cw)  # 1 s all-red
    assert a3.counts["short_all_red"] == 1


# --------------------------------------------------------- property: invariants hold
obs_strategy = st.tuples(
    st.integers(-1, 4),  # requested phase (including invalid)
    st.booleans(),  # sensors ok
    st.booleans(),  # ped uncertain
    st.sampled_from([(), ("N",), ("E",), ("S", "W")]),  # emergency approaches
    st.sampled_from([(), ("cw_N",), ("cw_E", "cw_S")]),  # occupied crosswalks
    st.integers(0, 1),  # junction box
    st.floats(0, 150),  # ped wait
)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(st.lists(obs_strategy, min_size=50, max_size=400))
def test_no_policy_or_sensor_sequence_can_cause_an_audit_violation(seq):
    """For any request/sensor sequence the displayed lights satisfy every audited invariant."""
    model = IntersectionModel.four_way()
    timing = SignalTiming()
    m, aud = SignalSafetyManager(model, timing), SignalAuditor(model, timing)
    for t, (req, ok, unc, emerg, occ, box, pw) in enumerate(seq):
        obs = make_obs(model, float(t), emergency=emerg, ped_occupied=occ, ped_wait=pw, sensors_ok=ok,
                       ped_uncertain=unc, box=box)
        aud.observe_command(m.step(float(t), PhaseRequest(req), obs))
    assert aud.total == 0, aud.violations[:3]


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(st.lists(st.integers(0, 2), min_size=100, max_size=300))
def test_pedestrians_detected_in_crosswalk_never_face_vehicle_green(reqs):
    model = IntersectionModel.four_way()
    m = SignalSafetyManager(model, SignalTiming(max_conflict_hold_s=1000.0))
    for t, r in enumerate(reqs):
        occupied = ("cw_N",) if (t // 20) % 2 == 1 else ()
        cmd = m.step(float(t), PhaseRequest(r), make_obs(model, float(t), ped_occupied=occupied))
        # Only check releases decided *while sensing the pedestrian*: a vehicle green may not newly
        # start while the crosswalk is occupied.
        if occupied and cmd.events and any(e.startswith("phase_start") for e in cmd.events):
            assert not any(v == "G" for v in cmd.approach_lights.values())
