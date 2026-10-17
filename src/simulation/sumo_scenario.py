"""SUMO network and demand generation (deterministic given a seed).

``build_network`` creates a single four-arm signalised intersection with sidewalks and four
zebra crossings using ``netconvert``. ``generate_routes`` writes a vehicle + pedestrian +
emergency-vehicle demand file whose content depends only on ``(scenario, seed)``.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

from ..errors import SimulationError

APPROACHES = ("N", "S", "E", "W")
_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
_LEFT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}  # destination arm for a left turn


# --------------------------------------------------------------------------- SUMO discovery
def find_sumo_home() -> Optional[Path]:
    """Locate SUMO: $SUMO_HOME, then the ``eclipse-sumo`` pip package, then PATH."""
    env = os.environ.get("SUMO_HOME")
    if env and Path(env).is_dir():
        return Path(env)
    try:  # pip install eclipse-sumo
        import sumo  # type: ignore

        home = Path(sumo.SUMO_HOME)
        if home.is_dir():
            return home
    except Exception:  # noqa: BLE001
        pass
    exe = shutil.which("sumo")
    if exe:
        return Path(exe).resolve().parent.parent
    return None


def find_binary(name: str) -> str:
    """Absolute path of a SUMO binary (``sumo``, ``sumo-gui``, ``netconvert``)."""
    home = find_sumo_home()
    candidates = []
    if home is not None:
        candidates += [home / "bin" / name, home / "bin" / f"{name}.exe"]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which(name)
    if found:
        return found
    raise SimulationError(
        f"SUMO binary {name!r} not found. Install SUMO (https://sumo.dlr.de) or `pip install eclipse-sumo`, "
        "and set SUMO_HOME. See README 'SUMO setup'."
    )


def sumo_available() -> bool:
    try:
        find_binary("sumo")
        import traci  # noqa: F401

        return True
    except (SimulationError, ImportError):
        return False


# --------------------------------------------------------------------------------- network
def build_network(out_dir: Union[str, Path], arm_length_m: float = 150.0, speed_mps: float = 13.89) -> Path:
    """Generate ``intersection.net.xml`` in ``out_dir`` (4 arms, 1 lane/direction, crossings)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L = arm_length_m
    pos = {"N": (0, L), "S": (0, -L), "E": (L, 0), "W": (-L, 0)}
    nodes = ['<nodes>', '  <node id="C" x="0" y="0" type="traffic_light"/>']
    nodes += [f'  <node id="{a}" x="{x}" y="{y}" type="priority"/>' for a, (x, y) in pos.items()]
    nodes.append("</nodes>")
    edges = ["<edges>"]
    for a in APPROACHES:
        edges.append(f'  <edge id="{a}_in" from="{a}" to="C" numLanes="1" speed="{speed_mps}"/>')
        edges.append(f'  <edge id="{a}_out" from="C" to="{a}" numLanes="1" speed="{speed_mps}"/>')
    edges.append("</edges>")
    (out / "intersection.nod.xml").write_text("\n".join(nodes), encoding="utf-8")
    (out / "intersection.edg.xml").write_text("\n".join(edges), encoding="utf-8")
    net = out / "intersection.net.xml"
    cmd = [
        find_binary("netconvert"),
        "-n", str(out / "intersection.nod.xml"), "-e", str(out / "intersection.edg.xml"),
        "-o", str(net), "--sidewalks.guess", "true", "--crossings.guess", "true",
        "--no-turnarounds", "true", "--no-warnings", "true",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not net.is_file():
        raise SimulationError(f"netconvert failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return net


# ------------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    """Demand description. Rates are per hour; ``profile`` multiplies vehicle demand over time."""

    name: str
    duration_s: float = 1800.0
    veh_per_hour: Union[float, Mapping[str, float]] = 350.0  # per approach
    ped_per_hour: Union[float, Mapping[str, float]] = 40.0  # per origin arm
    emergency_per_hour: float = 0.0  # total, random approach
    profile: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),)  # (start_s, multiplier)
    turn_probs: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # straight, left, right

    def rate(self, spec, arm: str) -> float:
        return float(spec[arm]) if isinstance(spec, Mapping) else float(spec)

    def multiplier(self, t: float) -> float:
        m = self.profile[0][1]
        for start, mult in self.profile:
            if t >= start:
                m = mult
        return m


# Single-lane arms with an exclusive pedestrian phase give roughly 1100 veh/h total capacity,
# so "medium" is ~0.8 utilisation and "high" is deliberately near / above saturation.
SCENARIOS: Dict[str, Scenario] = {
    "low": Scenario("low", veh_per_hour=140, ped_per_hour=20),
    "medium": Scenario("medium", veh_per_hour=220, ped_per_hour=40),
    "high": Scenario("high", veh_per_hour=290, ped_per_hour=60),
    "rush_hour": Scenario("rush_hour", veh_per_hour=240, ped_per_hour=50,
                          profile=((0, 0.6), (450, 1.2), (1050, 1.4), (1500, 0.7))),
    "asymmetric": Scenario("asymmetric", veh_per_hour={"N": 340, "S": 300, "E": 100, "W": 100}, ped_per_hour=40),
    "ped_heavy": Scenario("ped_heavy", veh_per_hour=220, ped_per_hour=150),
    "emergency": Scenario("emergency", veh_per_hour=220, ped_per_hour=40, emergency_per_hour=12.0),
}


def _poisson_times(rng: random.Random, rate_per_s: float, horizon: float, scenario: Scenario,
                   use_profile: bool) -> List[float]:
    """Thinned Poisson process with piecewise-constant multiplier."""
    if rate_per_s <= 0:
        return []
    peak = max(m for _, m in scenario.profile) if use_profile else 1.0
    lam = rate_per_s * peak
    t, out = 0.0, []  # type: ignore[var-annotated]
    while True:
        t += rng.expovariate(lam)
        if t >= horizon:
            return out
        if not use_profile or rng.random() < scenario.multiplier(t) / peak:
            out.append(t)


def generate_routes(scenario: Scenario, seed: int, out_path: Union[str, Path],
                    arm_length_m: float = 150.0) -> Path:
    """Write a SUMO route file; identical ``(scenario, seed)`` gives byte-identical output."""
    rng = random.Random(f"{scenario.name}:{seed}")
    out_path = Path(out_path)
    items: List[Tuple[float, str]] = []

    s_p, l_p, r_p = scenario.turn_probs
    for arm in APPROACHES:
        left_dest, right_dest = _LEFT_OF[arm], {v: k for k, v in _LEFT_OF.items()}[arm]
        for k, t in enumerate(_poisson_times(rng, scenario.rate(scenario.veh_per_hour, arm) / 3600.0,
                                             scenario.duration_s, scenario, True)):
            u = rng.random()
            dest = _OPPOSITE[arm] if u < s_p else left_dest if u < s_p + l_p else right_dest
            items.append((t, f'  <vehicle id="v_{arm}_{k}" type="car" depart="{t:.2f}" departLane="best" '
                             f'departSpeed="max"><route edges="{arm}_in {dest}_out"/></vehicle>'))
    for k, t in enumerate(_poisson_times(rng, scenario.emergency_per_hour / 3600.0, scenario.duration_s,
                                         scenario, False)):
        arm = rng.choice(APPROACHES)
        items.append((t, f'  <vehicle id="emg_{k}" type="emergency" depart="{t:.2f}" departLane="best" '
                         f'departSpeed="max"><route edges="{arm}_in {_OPPOSITE[arm]}_out"/></vehicle>'))
    for arm in APPROACHES:
        for k, t in enumerate(_poisson_times(rng, scenario.rate(scenario.ped_per_hour, arm) / 3600.0,
                                             scenario.duration_s, scenario, False)):
            dest = _OPPOSITE[arm] if rng.random() < 0.7 else rng.choice([a for a in APPROACHES if a != arm])
            pos = rng.uniform(80.0, max(81.0, arm_length_m - 15.0))
            items.append((t, f'  <person id="p_{arm}_{k}" depart="{t:.2f}" departPos="{pos:.1f}">'
                             f'<walk edges="{arm}_in {dest}_out"/></person>'))

    items.sort(key=lambda it: (it[0], it[1]))  # SUMO requires departure order
    header = [
        '<routes>',
        '  <vType id="car" vClass="passenger" length="4.5" accel="2.6" decel="4.5" tau="1.0" '
        'minGap="2.5" maxSpeed="13.89" sigma="0.5"/>',
        '  <vType id="emergency" vClass="emergency" length="6.5" accel="3.0" decel="4.5" tau="1.0" '
        'minGap="2.5" maxSpeed="19.44" speedFactor="1.1" color="1,0,0" guiShape="emergency"/>',
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + [x for _, x in items] + ["</routes>"]), encoding="utf-8")
    return out_path


# -------------------------------------------------------------------------------- validation
def validate_sumo_inputs(net_file: Union[str, Path], route_file: Union[str, Path], step_length: float) -> None:
    """Fail early (``SimulationError``) on missing / malformed inputs or bad parameters."""
    if not (0 < step_length <= 5):
        raise SimulationError(f"step_length must be in (0, 5], got {step_length}")
    for label, f in (("network", net_file), ("route", route_file)):
        p = Path(f)
        if not p.is_file():
            raise SimulationError(f"SUMO {label} file not found: {p}")
        try:
            ET.parse(p)
        except ET.ParseError as exc:
            raise SimulationError(f"SUMO {label} file {p} is not valid XML: {exc}") from exc
    root = ET.parse(net_file).getroot()
    if not any(j.get("type") == "traffic_light" for j in root.iter("junction")):
        raise SimulationError(f"network {net_file} has no traffic_light junction")
