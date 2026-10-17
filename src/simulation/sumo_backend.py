"""SUMO backend: runs the generated intersection under a ``SignalCommand``.

The backend only *displays* what the safety manager commands (it maps abstract lights to
the real TLS link indices read from the running network), reports what SUMO measures, and
independently reads back the executed TLS state so the auditor can verify what SUMO actually
shows.

Reproducibility: demand is regenerated from ``(scenario, seed)`` on every ``reset`` and SUMO
runs with ``--seed``; teleporting is disabled so gridlock stays visible in the metrics.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..errors import SimulationError
from ..safety.intersection import IntersectionModel
from ..state import (
    ApproachObservation,
    CrosswalkObservation,
    IntersectionObservation,
    SignalCommand,
)
from .backend import SensorNoise, StepStats, TrafficBackend
from .sumo_scenario import (
    SCENARIOS,
    Scenario,
    build_network,
    find_binary,
    generate_routes,
    validate_sumo_inputs,
)

try:
    import traci
    import traci.constants as tc

    TRACI_AVAILABLE = True
except ImportError:  # pragma: no cover
    traci = None
    tc = None
    TRACI_AVAILABLE = False

logger = logging.getLogger(__name__)

TLS_ID = "C"
_VEH_INTERNAL = re.compile(r"^:C_\d+_\d+$")
_STOPPED = 0.1  # m/s, SUMO's definition of "waiting"


class SumoBackend(TrafficBackend):
    """Single-intersection SUMO environment (see ``sumo_scenario.build_network``)."""

    def __init__(
        self,
        model: Optional[IntersectionModel] = None,
        scenario: Scenario | str = "medium",
        net_file: Optional[str] = None,
        step_length_s: float = 1.0,
        seed: int = 0,
        gui: bool = False,
        noise: Optional[SensorNoise] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        if not TRACI_AVAILABLE:
            raise SimulationError("traci is not installed: pip install -r requirements-sim.txt")
        model = model or IntersectionModel.four_way()
        super().__init__(model, noise, seed)
        if set(model.approaches) - {"N", "S", "E", "W"}:
            raise SimulationError("SumoBackend supports the generated N/S/E/W intersection only")
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.step_length_s = float(step_length_s)
        self._seed = seed
        self._gui = gui
        self._owns_dir = work_dir is None
        self._dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="sumo_tls_"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._net = Path(net_file) if net_file else build_network(self._dir)
        self._label = f"tls_{uuid.uuid4().hex}"  # unique: id() values are reused after GC
        self._conn: Any = None
        self._t = 0.0
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    @property
    def time_s(self) -> float:
        return self._t

    def _sumo_args(self, routes: Path) -> List[str]:
        return [
            "-n", str(self._net), "-r", str(routes),
            "--step-length", str(self.step_length_s), "--seed", str(self._seed),
            "--no-step-log", "true", "--no-warnings", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1", "--max-depart-delay", "-1",
            "--collision.action", "warn",
        ]

    def reset(self, seed: Optional[int] = None) -> None:
        if self._closed:
            raise SimulationError("backend is closed")
        if seed is not None:
            self._seed = seed
        self._noise_rng.seed(self._seed + 7919)
        routes = generate_routes(self.scenario, self._seed, self._dir / "routes.rou.xml")
        validate_sumo_inputs(self._net, routes, self.step_length_s)
        self._planned_departs = sorted(
            float(v.get("depart", 0)) for v in ET.parse(routes).getroot().iter("vehicle")
        )
        args = self._sumo_args(routes)
        try:
            if self._conn is None:
                binary = find_binary("sumo-gui" if self._gui else "sumo")
                traci.start([binary] + args, label=self._label)
                self._conn = traci.getConnection(self._label)
            else:
                self._conn.load(args)
            self._init_episode()
        except SimulationError:
            self._abort()
            raise
        except Exception as exc:  # traci.FatalTraCIError, OSError, ...
            self._abort()
            raise SimulationError(f"failed to start/reset SUMO: {exc}") from exc

    def _init_episode(self) -> None:
        c = self._conn
        self._t = 0.0
        if TLS_ID not in c.trafficlight.getIDList():
            raise SimulationError(f"traffic light {TLS_ID!r} not found in network")
        c.trafficlight.setPhaseDuration(TLS_ID, 1e5)  # we drive the state every step
        self._build_link_map()
        self._lanes = {}
        for a in self.model.approaches:
            lanes = [f"{a}_in_{i}" for i in range(c.edge.getLaneNumber(f"{a}_in"))]
            veh = [ln for ln in lanes if not c.lane.getAllowed(ln) or "pedestrian" not in c.lane.getAllowed(ln)]
            if not veh:
                raise SimulationError(f"no vehicle lane on edge {a}_in")
            self._lanes[a] = veh[0]
        self._internal_lanes = [ln for ln in c.lane.getIDList() if _VEH_INTERNAL.match(ln)]
        for ln in list(self._lanes.values()) + self._internal_lanes:
            c.lane.subscribe(ln, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER, tc.LAST_STEP_VEHICLE_NUMBER,
                                  tc.LAST_STEP_VEHICLE_ID_LIST])
        c.simulation.subscribe([tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS,
                                tc.VAR_COLLIDING_VEHICLES_NUMBER, tc.VAR_EMERGENCYSTOPPING_VEHICLES_NUMBER])
        # per-episode accounting
        self._veh_depart: Dict[str, float] = {}
        self._veh_wait: Dict[str, float] = {}  # latest accumulated waiting time
        self._veh_done_wait: List[float] = []
        self._veh_travel: List[float] = []
        self._emg_wait: Dict[str, float] = {}
        self._emg_travel: List[float] = []
        self._emg_wait_done: List[float] = []
        self._ped_wait: Dict[str, float] = {}
        self._known_persons: Set[str] = set()
        self._departed_total = 0
        self._arrived_total = 0
        self._crossed_total = 0
        self._queue_sum = 0.0
        self._steps = 0
        self._collisions = 0
        self._emergency_stops = 0
        self._tls_mismatch = 0
        self._prev_lane_vehicles: Dict[str, Set[str]] = {a: set() for a in self.model.approaches}
        # Nothing has happened yet; do not read subscription results (they may still hold the
        # previous episode's departed/arrived ids right after ``load``).
        self._latest = IntersectionObservation(
            time_s=0.0,
            approaches={a: ApproachObservation() for a in self.model.approaches},
            crosswalks={cw: CrosswalkObservation() for cw in self.model.crosswalks},
        )

    def _build_link_map(self) -> None:
        """Map every TLS link index to an approach (+turn direction) or a crosswalk."""
        c = self._conn
        links = c.trafficlight.getControlledLinks(TLS_ID)
        root = ET.parse(self._net).getroot()
        crossing_arm: Dict[str, str] = {}
        for e in root.iter("edge"):
            if e.get("function") == "crossing":
                crossing_arm[e.get("id", "")] = e.get("crossingEdges", "").split()[0].split("_")[0]
        self._crossing_cw = {eid: f"cw_{arm}" for eid, arm in crossing_arm.items()}
        self._n_links = len(links)
        self._link_elem: List[Tuple[str, str]] = []  # (kind, element) kind in {"approach-s","approach-l","crossing"}
        direction: Dict[Tuple[str, str], str] = {}
        for a in self.model.approaches:
            for ln in (f"{a}_in_{i}" for i in range(c.edge.getLaneNumber(f"{a}_in"))):
                for entry in c.lane.getLinks(ln):
                    direction[(ln, entry[4])] = entry[6]
        for group in links:
            if not group:
                raise SimulationError("empty TLS link group")
            in_lane, out_lane, via = group[0]
            if in_lane.startswith(":"):  # walking area -> crossing lane
                edge = out_lane.rsplit("_", 1)[0]
                cw = self._crossing_cw.get(edge)
                if cw is None:
                    raise SimulationError(f"cannot map TLS link to {out_lane!r} onto a crosswalk")
                self._link_elem.append(("crossing", cw))
            else:
                arm = in_lane.split("_in")[0]
                d = direction.get((in_lane, via), "s")
                self._link_elem.append(("approach-l" if d == "l" else "approach", arm))
        missing = set(self.model.crosswalks) - {e for k, e in self._link_elem if k == "crossing"}
        if missing:
            raise SimulationError(f"network has no crossing for model crosswalks {sorted(missing)}")

    def _state_string(self, cmd: SignalCommand) -> str:
        out = []
        for kind, elem in self._link_elem:
            if kind == "crossing":
                out.append("G" if cmd.crosswalk_lights.get(elem) == "W" else "r")
            else:
                light = cmd.approach_lights.get(elem, "r")
                if light == "G":
                    out.append("g" if kind == "approach-l" else "G")  # permitted left yields
                else:
                    out.append("y" if light == "y" else "r")
        return "".join(out)

    # ------------------------------------------------------------------- stepping
    def step(self, cmd: SignalCommand) -> StepStats:
        c = self._conn
        if c is None:
            raise SimulationError("reset() must be called before step()")
        stats = StepStats()
        try:
            sent = self._state_string(cmd)
            c.trafficlight.setRedYellowGreenState(TLS_ID, sent)
            c.simulationStep()
            if c.trafficlight.getRedYellowGreenState(TLS_ID) != sent:
                self._tls_mismatch += 1  # SUMO displays something other than what was commanded
            self._t = float(c.simulation.getTime())
            self._latest = self._read_state(stats, cmd)
        except SimulationError:
            self._abort()
            raise
        except Exception as exc:  # connection lost / SUMO crashed
            self._abort()
            raise SimulationError(f"SUMO connection failed during step: {exc}") from exc
        return stats

    def executed_state(self) -> str:
        """TLS state string SUMO actually has (for independent auditing)."""
        return self._conn.trafficlight.getRedYellowGreenState(TLS_ID)

    def _read_state(self, stats: StepStats, cmd: Optional[SignalCommand]):
        c, dt, m = self._conn, self.step_length_s, self.model
        sim = c.simulation.getSubscriptionResults()
        lane_res = c.lane.getAllSubscriptionResults()

        # new / finished vehicles
        for vid in sim.get(tc.VAR_DEPARTED_VEHICLES_IDS, ()):
            self._veh_depart[vid] = self._t
            self._departed_total += 1
            try:
                c.vehicle.subscribe(vid, [tc.VAR_ROAD_ID, tc.VAR_SPEED, tc.VAR_WAITING_TIME,
                                          tc.VAR_ACCUMULATED_WAITING_TIME])
            except traci.TraCIException:  # departed and already gone within the same step
                pass
        veh_res = c.vehicle.getAllSubscriptionResults()
        for vid, r in veh_res.items():
            self._veh_wait[vid] = r[tc.VAR_ACCUMULATED_WAITING_TIME]
            if vid.startswith("emg_"):
                self._emg_wait[vid] = r[tc.VAR_ACCUMULATED_WAITING_TIME]
        for vid in sim.get(tc.VAR_ARRIVED_VEHICLES_IDS, ()):
            self._arrived_total += 1
            wait = self._veh_wait.get(vid, 0.0)
            travel = self._t - self._veh_depart.get(vid, self._t)
            self._veh_done_wait.append(wait)
            self._veh_travel.append(travel)
            if vid.startswith("emg_"):
                self._emg_travel.append(travel)
                self._emg_wait_done.append(wait)
                self._emg_wait.pop(vid, None)
            self._veh_wait.pop(vid, None)
            self._veh_depart.pop(vid, None)
        self._collisions += int(sim.get(tc.VAR_COLLIDING_VEHICLES_NUMBER, 0))
        self._emergency_stops += int(sim.get(tc.VAR_EMERGENCYSTOPPING_VEHICLES_NUMBER, 0))

        # approaches
        approaches: Dict[str, ApproachObservation] = {}
        total_halting = 0
        for a, ln in self._lanes.items():
            lr = lane_res[ln]
            ids = lr[tc.LAST_STEP_VEHICLE_ID_LIST]
            halting = int(lr[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            total_halting += halting
            waits = [veh_res[v][tc.VAR_WAITING_TIME] for v in ids if v in veh_res]
            stopped = [veh_res[v][tc.VAR_WAITING_TIME] for v in ids
                       if v in veh_res and veh_res[v][tc.VAR_SPEED] < _STOPPED]
            approaches[a] = ApproachObservation(
                queue=float(halting),
                max_wait_s=float(max(waits, default=0.0)),
                mean_wait_s=float(np.mean(stopped)) if stopped else 0.0,
                emergency=any(v.startswith("emg_") for v in ids),
            )
            prev = self._prev_lane_vehicles[a]
            cur = set(ids)
            left = len(prev - cur)
            stats.discharged += left
            self._crossed_total += left
            self._prev_lane_vehicles[a] = cur
        stats.vehicle_wait_s = total_halting * dt
        stats.emergency_wait_s = sum(
            dt for ln in self._lanes.values() for v in lane_res[ln][tc.LAST_STEP_VEHICLE_ID_LIST]
            if v.startswith("emg_") and veh_res.get(v, {}).get(tc.VAR_SPEED, 1.0) < _STOPPED
        )
        box = sum(int(lane_res[ln][tc.LAST_STEP_VEHICLE_NUMBER]) for ln in self._internal_lanes if ln in lane_res)

        # pedestrians
        pids = set(c.person.getIDList())
        for pid in pids - self._known_persons:
            try:
                c.person.subscribe(pid, [tc.VAR_ROAD_ID, tc.VAR_SPEED, tc.VAR_WAITING_TIME])
            except traci.TraCIException:
                continue
            self._ped_wait.setdefault(pid, 0.0)
        self._known_persons = pids
        per_res = c.person.getAllSubscriptionResults()
        waiting_n = {cw: 0 for cw in m.crosswalks}
        waiting_max = {cw: 0.0 for cw in m.crosswalks}
        occupied = {cw: 0 for cw in m.crosswalks}
        ped_wait_total = 0
        for pid, r in per_res.items():
            road, speed = r[tc.VAR_ROAD_ID], r[tc.VAR_SPEED]
            if speed < _STOPPED:
                self._ped_wait[pid] = self._ped_wait.get(pid, 0.0) + dt
                ped_wait_total += 1
            if road in self._crossing_cw:
                occupied[self._crossing_cw[road]] += 1
            elif speed < _STOPPED and road.startswith(":"):
                cw = self._waiting_for(pid)
                if cw in waiting_n:
                    waiting_n[cw] += 1
                    waiting_max[cw] = max(waiting_max[cw], r[tc.VAR_WAITING_TIME])
        stats.pedestrian_wait_s = ped_wait_total * dt
        crosswalks = {
            cw: CrosswalkObservation(float(waiting_n[cw]), float(waiting_max[cw]), float(occupied[cw]))
            for cw in m.crosswalks
        }
        if cmd is not None:
            for cw, n in occupied.items():
                if n and any(m.conflict(cw, a) and cmd.approach_lights.get(a, "r") in ("G", "y")
                             for a in m.approaches):
                    stats.ped_conflict = True
                    break
            self._queue_sum += total_halting
            self._steps += 1
        return IntersectionObservation(time_s=self._t, approaches=approaches, crosswalks=crosswalks,
                                       box_vehicles=int(box))

    def _waiting_for(self, pid: str) -> Optional[str]:
        try:
            nxt = self._conn.person.getNextEdge(pid)
        except Exception:  # noqa: BLE001
            return None
        return self._crossing_cw.get(nxt)

    # ---------------------------------------------------------------- observation
    def _truth(self) -> IntersectionObservation:
        return self._latest

    def episode_metrics(self) -> Dict[str, float]:
        waits = self._veh_done_wait + list(self._veh_wait.values())
        ped = list(self._ped_wait.values())
        emg_wait = self._emg_wait_done + list(self._emg_wait.values())
        backlog = sum(1 for d in self._planned_departs if d <= self._t) - self._departed_total
        return {
            "mean_vehicle_wait_s": float(np.mean(waits)) if waits else 0.0,
            "max_vehicle_wait_s": float(max(waits, default=0.0)),
            "mean_travel_time_s": float(np.mean(self._veh_travel)) if self._veh_travel else 0.0,
            "mean_pedestrian_wait_s": float(np.mean(ped)) if ped else 0.0,
            "max_pedestrian_wait_s": float(max(ped, default=0.0)),
            "mean_queue": self._queue_sum / max(1, self._steps),
            "throughput_veh": float(self._crossed_total),
            "arrived_veh": float(self._arrived_total),
            "mean_emergency_delay_s": float(np.mean(emg_wait)) if emg_wait else 0.0,
            "mean_emergency_travel_s": float(np.mean(self._emg_travel)) if self._emg_travel else 0.0,
            "emergency_vehicles_served": float(len(self._emg_travel)),
            "collisions": float(self._collisions),
            "tls_state_mismatch_steps": float(self._tls_mismatch),
            "emergency_brakes": float(self._emergency_stops),
            "insertion_backlog": float(max(0, backlog)),
            "pedestrians": float(len(ped)),
        }

    # -------------------------------------------------------------------- cleanup
    def _abort(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - process may already be dead
                pass
            try:  # a failed close can leave the label registered in traci's global table
                traci.main._connections.pop(self._label, None)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._abort()
        if self._owns_dir:
            shutil.rmtree(self._dir, ignore_errors=True)
