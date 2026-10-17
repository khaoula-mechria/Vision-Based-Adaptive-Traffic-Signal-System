"""Intersection topology: approaches, crosswalks, phases and the conflict relation.

The conflict relation is *data*, validated at construction. A phase that gives green /
walk to two conflicting elements is rejected, so an invalid phase plan can never reach the
signal executor regardless of what any policy (RL or otherwise) requests.

Elements are plain strings: vehicle approaches ("N", "S", ...) and crosswalks ("cw_N", ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, FrozenSet, List, Mapping, Optional, Tuple

from ..errors import IntersectionModelError


@dataclass(frozen=True)
class PhaseDef:
    """A signal phase: which approaches get green and which crosswalks get walk."""

    name: str
    green_approaches: FrozenSet[str] = frozenset()
    walk_crosswalks: FrozenSet[str] = frozenset()

    @property
    def is_pedestrian(self) -> bool:
        return bool(self.walk_crosswalks)

    @property
    def elements(self) -> FrozenSet[str]:
        return self.green_approaches | self.walk_crosswalks


@dataclass(frozen=True)
class IntersectionModel:
    approaches: Tuple[str, ...]
    crosswalks: Tuple[str, ...]
    phases: Tuple[PhaseDef, ...]
    conflicts: FrozenSet[FrozenSet[str]]

    def __post_init__(self) -> None:
        self.validate()

    # ---- validation -------------------------------------------------------------
    def validate(self) -> None:
        elements = set(self.approaches) | set(self.crosswalks)
        if len(elements) != len(self.approaches) + len(self.crosswalks):
            raise IntersectionModelError("approach and crosswalk names must be unique")
        if not self.approaches:
            raise IntersectionModelError("at least one approach is required")
        if not self.phases:
            raise IntersectionModelError("at least one phase is required")
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise IntersectionModelError(f"duplicate phase names: {names}")
        for pair in self.conflicts:
            if len(pair) != 2 or not pair <= elements:
                raise IntersectionModelError(f"invalid conflict pair {set(pair)}")
        for phase in self.phases:
            unknown = phase.elements - elements
            if unknown:
                raise IntersectionModelError(f"phase {phase.name!r} references unknown elements {sorted(unknown)}")
            if not phase.elements:
                raise IntersectionModelError(f"phase {phase.name!r} serves nothing")
            for a, b in combinations(sorted(phase.elements), 2):
                if frozenset((a, b)) in self.conflicts:
                    raise IntersectionModelError(
                        f"phase {phase.name!r} gives simultaneous service to conflicting elements {a!r} and {b!r}"
                    )
        served = set().union(*(p.green_approaches for p in self.phases))
        starved = set(self.approaches) - served
        if starved:
            raise IntersectionModelError(f"approaches never served by any phase: {sorted(starved)}")
        served_cw = set().union(*(p.walk_crosswalks for p in self.phases))
        if set(self.crosswalks) - served_cw:
            raise IntersectionModelError(
                f"crosswalks never served by any phase: {sorted(set(self.crosswalks) - served_cw)}"
            )

    # ---- queries ----------------------------------------------------------------
    @property
    def num_phases(self) -> int:
        return len(self.phases)

    def conflict(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.conflicts

    def phases_conflict(self, p: int, q: int) -> bool:
        """True if any element of phase p conflicts with any element of phase q."""
        return any(self.conflict(a, b) for a in self.phases[p].elements for b in self.phases[q].elements)

    def phase_for_approach(self, approach: str) -> Optional[int]:
        for i, ph in enumerate(self.phases):
            if approach in ph.green_approaches:
                return i
        return None

    def pedestrian_phase(self) -> Optional[int]:
        for i, ph in enumerate(self.phases):
            if ph.is_pedestrian:
                return i
        return None

    def crosswalks_conflicting_with_phase(self, phase: int) -> List[str]:
        """Crosswalks that must be empty before ``phase`` may turn green."""
        ph = self.phases[phase]
        return [
            cw for cw in self.crosswalks
            if cw not in ph.walk_crosswalks and any(self.conflict(cw, a) for a in ph.green_approaches)
        ]

    # ---- construction -----------------------------------------------------------
    @classmethod
    def four_way(cls, with_pedestrian_phase: bool = True) -> "IntersectionModel":
        """Conservative default: NS / EW vehicle phases and an exclusive pedestrian phase.

        Every crosswalk conflicts with every approach, so pedestrians never share time
        with vehicles (a deliberate safety-over-efficiency choice; see README).
        """
        approaches = ("N", "S", "E", "W")
        crosswalks = tuple(f"cw_{a}" for a in approaches)
        conflicts = {frozenset((a, b)) for a in ("N", "S") for b in ("E", "W")}
        if with_pedestrian_phase:
            conflicts |= {frozenset((cw, a)) for cw in crosswalks for a in approaches}
        phases = [
            PhaseDef("NS", frozenset({"N", "S"})),
            PhaseDef("EW", frozenset({"E", "W"})),
        ]
        if with_pedestrian_phase:
            phases.append(PhaseDef("PED", frozenset(), frozenset(crosswalks)))
        return cls(approaches, crosswalks if with_pedestrian_phase else (), tuple(phases), frozenset(conflicts))

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "IntersectionModel":
        """Build from YAML-style config; see ``config/default_config.yaml`` (signal.intersection)."""
        try:
            approaches = tuple(cfg["approaches"])
            crosswalks = tuple(cfg.get("crosswalks", ()))
            phases = tuple(
                PhaseDef(
                    str(p["name"]),
                    frozenset(p.get("green", ())),
                    frozenset(p.get("walk", ())),
                )
                for p in cfg["phases"]
            )
            conflicts = frozenset(frozenset(pair) for pair in cfg["conflicts"])
        except (KeyError, TypeError) as exc:
            raise IntersectionModelError(f"malformed intersection config: {exc!r}") from exc
        return cls(approaches, crosswalks, phases, conflicts)
