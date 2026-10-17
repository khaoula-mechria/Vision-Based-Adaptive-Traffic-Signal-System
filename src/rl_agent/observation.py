"""Observation (state) encoding for the RL agent.

Vector layout, for A approaches, C crosswalks and P phases (default 4, 4, 3 -> dim 25):

    [ queue_a / max_queue                    ] * A   halting vehicles per approach
    [ max_wait_a / max_wait                  ] * A   longest wait per approach
    [ emergency_a                            ] * A   0/1 confirmed emergency vehicle
    [ waiting_c / max_ped                    ] * C   pedestrians waiting per crosswalk
    [ ped_max_wait_c / max_wait              ] * C   longest pedestrian wait per crosswalk
    [ one_hot(current phase)                 ] * P
    [ phase_elapsed / max_phase_time, in_transition ]

All entries are clipped to [0, 1]. Dimensions are derived from the ``IntersectionModel``;
nothing is hard coded to four approaches.
"""

from __future__ import annotations

import numpy as np

from ..safety.intersection import IntersectionModel
from ..safety.signal_safety import PhaseContext
from ..state import IntersectionObservation, Stage


class ObservationEncoder:
    def __init__(
        self,
        model: IntersectionModel,
        max_queue: float = 30.0,
        max_wait_s: float = 120.0,
        max_ped: float = 20.0,
        max_phase_time_s: float = 60.0,
    ) -> None:
        self.model = model
        self.max_queue = max_queue
        self.max_wait_s = max_wait_s
        self.max_ped = max_ped
        self.max_phase_time_s = max_phase_time_s
        a, c, p = len(model.approaches), len(model.crosswalks), model.num_phases
        self.dim = 3 * a + 2 * c + p + 2

    def encode(self, obs: IntersectionObservation, ctx: PhaseContext) -> np.ndarray:
        m = self.model
        v = []
        for name in m.approaches:
            v.append(obs.approaches[name].queue / self.max_queue if name in obs.approaches else 0.0)
        for name in m.approaches:
            v.append(obs.approaches[name].max_wait_s / self.max_wait_s if name in obs.approaches else 0.0)
        for name in m.approaches:
            v.append(float(obs.approaches[name].emergency) if name in obs.approaches else 0.0)
        for name in m.crosswalks:
            v.append(obs.crosswalks[name].waiting / self.max_ped if name in obs.crosswalks else 0.0)
        for name in m.crosswalks:
            v.append(obs.crosswalks[name].max_wait_s / self.max_wait_s if name in obs.crosswalks else 0.0)
        onehot = [0.0] * m.num_phases
        onehot[ctx.target_phase if ctx.stage is not Stage.HOLD else ctx.phase] = 1.0
        v.extend(onehot)
        v.append(ctx.phase_elapsed_s / self.max_phase_time_s)
        v.append(0.0 if ctx.stage is Stage.HOLD else 1.0)
        arr = np.asarray(v, dtype=np.float32)
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
