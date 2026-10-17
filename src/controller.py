"""Adaptive traffic signal controller: wires the stages together.

    frame -> perceive (detect, track) -> estimate (lanes, crosswalks, emergency)
          -> IntersectionObservation -> decide (policy -> PhaseRequest)
          -> safety manager (validates, performs transitions) -> SignalCommand -> output

Each stage has one responsibility and exchanges the dataclasses in ``state.py``. The policy
(RL or otherwise) can only *request* a phase; ``SignalSafetyManager`` is the sole producer of
lights. Perception failures never stop the signal: after repeated failures or stale data the
observation is marked ``sensors_ok=False`` and the safety manager runs its fixed fallback cycle.

Research prototype - not certified for control of public-road traffic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from .control.policies import ActuatedPolicy, FixedTimePolicy, GuardedPolicy, RLPolicy, SignalPolicy
from .emergency import EmergencyVehicleHandler
from .errors import ConfigError, DetectionError, ModelNotFoundError
from .perception.postprocess import FilterConfig
from .queue_estimation import QueueEstimator
from .safety import CrosswalkMonitor, IntersectionModel, SignalAuditor, SignalSafetyManager, SignalTiming
from .state import (
    ApproachObservation,
    CrosswalkObservation,
    IntersectionObservation,
    PhaseRequest,
    SignalCommand,
)
from .tracking import MultiObjectTracker
from .utils import ConfigLoader, MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    timestamp: float
    command: SignalCommand
    observation: IntersectionObservation
    request: Optional[PhaseRequest]
    sensors_ok: bool
    num_detections: int = 0
    num_tracks: int = 0
    processing_ms: float = 0.0
    error: Optional[str] = None
    audit_violations: int = 0


class AdaptiveTrafficController:
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[ConfigLoader] = None,
        detector: Any = None,
        policy: Optional[SignalPolicy] = None,
        model: Optional[IntersectionModel] = None,
        on_command: Optional[Callable[[SignalCommand], None]] = None,
    ) -> None:
        self.config = config or ConfigLoader(config_path)
        cfg = self.config
        self.model = model or self._build_model()
        self.timing = SignalTiming.from_dict(cfg.get("signal.timing"))
        self.manager = SignalSafetyManager(self.model, self.timing)
        self.auditor = SignalAuditor(self.model, self.timing)

        pc = cfg.get_section("perception")
        self.detector = detector if detector is not None else self._build_detector(pc)
        tc = cfg.get_section("tracking")
        self.tracker = MultiObjectTracker(tc["max_age"], tc["min_hits"], tc["iou_threshold"],
                                          tc["output_coasting"], tc["max_gap_s"])
        qc = cfg.get_section("queue_estimation")
        self.queue = QueueEstimator.from_config(qc)
        self.crosswalks = CrosswalkMonitor.from_config(cfg.get("crosswalks") or [])
        ec = cfg.get_section("emergency")
        self.emergency = EmergencyVehicleHandler(ec["detection_confidence"], ec["min_detections"], ec["hold_s"],
                                                 approach_of=self._approach_of)
        self.policy = GuardedPolicy(policy or self._build_policy(), FixedTimePolicy(self.model))
        self.metrics = MetricsCollector()
        self.on_command = on_command

        self._max_failures = pc["max_consecutive_failures"]
        self._max_stale = pc["max_stale_s"]
        self._failures = 0
        self._last_good_ts: Optional[float] = None
        self._last_obs = self._empty_obs(0.0, sensors_ok=False)
        self.frame_count = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ construction
    def _build_model(self) -> IntersectionModel:
        spec = self.config.get("signal.intersection")
        return IntersectionModel.from_dict(spec) if spec else IntersectionModel.four_way()

    def _build_detector(self, pc: Dict[str, Any]):
        fc = FilterConfig(pc["confidence_threshold"], min_box_area_px=pc["min_box_area_px"],
                          emergency_confidence=self.config.get("emergency.detection_confidence"))
        if pc["backend"] == "onnx":
            from .edge import OnnxYoloDetector

            return OnnxYoloDetector(pc["onnx_path"], filter_config=fc, input_size=pc["input_size"],
                                    nms_threshold=pc["nms_threshold"])
        from .perception import ObjectDetector

        return ObjectDetector(pc["model_name"], pc["model_path"], filter_config=fc, nms_threshold=pc["nms_threshold"],
                              device=pc["device"], input_size=(pc["input_size"],) * 2,
                              allow_download=pc["allow_download"])

    def _build_policy(self) -> SignalPolicy:
        name = self.config.get("signal.policy")
        if name == "fixed":
            return FixedTimePolicy(self.model)
        if name == "rl":
            try:
                from .rl_agent import ObservationEncoder
                from .rl_agent.agent import TrafficSignalAgent

                encoder = ObservationEncoder(self.model)
                agent = TrafficSignalAgent.load(self.config.get("signal.rl_checkpoint"))
                agent.check_compatible(encoder.dim, self.model.num_phases)
                return RLPolicy(agent, encoder)
            except (ModelNotFoundError, ConfigError, ImportError) as exc:
                logger.error("RL policy unavailable (%s); using the actuated baseline instead", exc)
        return ActuatedPolicy(self.model)

    def _approach_of(self, track) -> Optional[str]:
        lane = self.queue._assign(*track.get_center())
        return lane.approach if lane else None

    # ------------------------------------------------------------------------ stages
    def _empty_obs(self, ts: float, sensors_ok: bool) -> IntersectionObservation:
        return IntersectionObservation(
            time_s=ts,
            approaches={a: ApproachObservation() for a in self.model.approaches},
            crosswalks={c: CrosswalkObservation() for c in self.model.crosswalks},
            sensors_ok=sensors_ok,
        )

    def perceive(self, frame: Optional[np.ndarray], ts: float):
        """Detect + track. Returns ``(detections, tracks)`` or ``None`` when the frame is unusable."""
        try:
            result = self.detector.detect(frame, ts)
        except DetectionError as exc:
            self._failures += 1
            self.last_error = str(exc)
            logger.warning("perception failure %d: %s", self._failures, exc)
            return None
        except Exception as exc:  # noqa: BLE001 - e.g. missing model file, runtime crash
            self._failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error("perception error %d: %s", self._failures, self.last_error)
            return None
        self._failures = 0
        self._last_good_ts = ts
        self.metrics.record_latency(result.inference_time_ms, "inference")
        return result, self.tracker.update(result)

    def estimate(self, tracks, ts: float) -> IntersectionObservation:
        self.queue.update(tracks, ts)
        agg = self.queue.approach_states()
        emergency = self.emergency.update(tracks, ts)
        cw_obs, ped_uncertain = self.crosswalks.update(tracks, ts) if self.crosswalks.zones else ({}, False)
        approaches = {}
        for a in self.model.approaches:
            s = agg.get(a, {})
            approaches[a] = ApproachObservation(
                queue=float(s.get("queue", 0.0)), max_wait_s=float(s.get("max_wait_s", 0.0)),
                mean_wait_s=float(s.get("mean_wait_s", 0.0)), emergency=a in emergency)
        crosswalks = {c: cw_obs.get(c, CrosswalkObservation()) for c in self.model.crosswalks}
        # A model with crosswalks but no configured zones means pedestrians are not sensed at all.
        unsensed = bool(self.model.crosswalks) and not self.crosswalks.zones
        return IntersectionObservation(ts, approaches, crosswalks, sensors_ok=True,
                                       ped_uncertain=ped_uncertain or unsensed)

    # ------------------------------------------------------------------------- loop
    def step(self, frame: Optional[np.ndarray], timestamp: Optional[float] = None) -> StepResult:
        """Process one camera frame (``None`` = missing frame) and return the signal command."""
        ts = time.time() if timestamp is None else float(timestamp)
        t0 = time.perf_counter()
        self.frame_count += 1
        out = self.perceive(frame, ts)
        n_det = n_trk = 0
        if out is not None:
            result, tracks = out
            n_det, n_trk = len(result.detections), len(tracks)
            obs = self.estimate(tracks, ts)
        else:
            obs = self._last_obs
            obs = IntersectionObservation(ts, obs.approaches, obs.crosswalks, sensors_ok=obs.sensors_ok,
                                          ped_uncertain=True)
        stale = self._last_good_ts is None or ts - self._last_good_ts > self._max_stale
        if self._failures >= self._max_failures or stale:
            obs = IntersectionObservation(ts, obs.approaches, obs.crosswalks, sensors_ok=False, ped_uncertain=True)
        self._last_obs = obs
        return self.tick(obs, ts, n_det, n_trk, t0)

    def tick(self, obs: IntersectionObservation, ts: float, n_det: int = 0, n_trk: int = 0,
             t0: Optional[float] = None) -> StepResult:
        """Decide + execute for an observation (also usable without a camera, e.g. in tests)."""
        t0 = time.perf_counter() if t0 is None else t0
        ctx = self.manager.context(ts)
        try:
            request = self.policy.request(obs, ctx)
        except Exception as exc:  # noqa: BLE001 - GuardedPolicy already catches; last line of defence
            logger.exception("decision stage failed")
            request = None
            self.last_error = str(exc)
        cmd = self.manager.step(ts, request, obs)
        violations = self.auditor.observe_command(cmd)
        if violations:
            logger.critical("AUDIT VIOLATION: %s", violations)
        for ev in cmd.events:
            logger.info("signal event: %s", ev)
        if self.on_command:
            self.on_command(cmd)
        ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.record_latency(ms, "frame_processing")
        return StepResult(ts, cmd, obs, request, obs.sensors_ok, n_det, n_trk, ms, self.last_error, len(violations))

    def get_status(self) -> Dict[str, Any]:
        return {
            "phase": self.model.phases[self.manager.phase].name,
            "stage": self.manager.stage.value,
            "policy": self.policy.name,
            "policy_fallbacks": self.policy.fallback_count,
            "frames": self.frame_count,
            "consecutive_perception_failures": self._failures,
            "last_error": self.last_error,
            "safety_stats": dict(self.manager.stats),
            "audit_violations": self.auditor.total,
            "queue": self.queue.get_summary()["total_queue"],
            "tracking": self.tracker.get_statistics(),
        }

    def reset(self) -> None:
        self.tracker.reset()
        self.queue.reset()
        self.crosswalks.reset()
        self.emergency.reset()
        self.manager = SignalSafetyManager(self.model, self.timing)
        self.auditor = SignalAuditor(self.model, self.timing)
        self.policy.reset()
        self._failures, self._last_good_ts, self.frame_count = 0, None, 0
