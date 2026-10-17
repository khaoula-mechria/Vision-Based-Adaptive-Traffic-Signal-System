"""Shared fixtures and helpers."""

import pytest

from src.perception.detection_result import BoundingBox, Detection, DetectionResult
from src.safety import IntersectionModel, SignalSafetyManager, SignalTiming
from src.state import (
    ApproachObservation,
    CrosswalkObservation,
    IntersectionObservation,
)


@pytest.fixture
def model():
    return IntersectionModel.four_way()


@pytest.fixture
def timing():
    return SignalTiming()


@pytest.fixture
def manager(model, timing):
    return SignalSafetyManager(model, timing)


def make_obs(model, t, queues=None, emergency=(), ped_occupied=(), ped_wait=0.0, ped_waiting=0,
             sensors_ok=True, ped_uncertain=False, box=0):
    queues = queues or {}
    return IntersectionObservation(
        time_s=t,
        approaches={a: ApproachObservation(queue=queues.get(a, 0), max_wait_s=queues.get(a, 0) * 2.0,
                                           emergency=a in emergency) for a in model.approaches},
        crosswalks={c: CrosswalkObservation(waiting=ped_waiting, max_wait_s=ped_wait,
                                            occupied=1 if c in ped_occupied else 0) for c in model.crosswalks},
        sensors_ok=sensors_ok, ped_uncertain=ped_uncertain, box_vehicles=box,
    )


def run(manager, model, seconds, request=None, obs_fn=None, start=0.0, auditor=None):
    """Step the manager once per second; returns the list of commands."""
    cmds = []
    for k in range(int(seconds)):
        t = start + k
        obs = obs_fn(t) if obs_fn else make_obs(model, t)
        req = request(t) if callable(request) else request
        cmd = manager.step(t, req, obs)
        if auditor:
            auditor.observe_command(cmd)
        cmds.append(cmd)
    return cmds


def detection(x1, y1, x2, y2, cls="car", conf=0.9, **flags):
    kind = dict(is_vehicle=cls in ("car", "bus", "truck", "motorcycle", "ambulance"),
                is_pedestrian=cls == "person", is_cyclist=cls == "bicycle", is_emergency=cls == "ambulance")
    kind.update(flags)
    return Detection(BoundingBox(x1, y1, x2, y2), 0, cls, conf, **kind)


def det_result(ts, dets, frame_id=0):
    return DetectionResult(frame_id=frame_id, timestamp=ts, detections=list(dets))
