# Vision-Based Adaptive Traffic Signal System

> **Research prototype. Not certified, validated or approved for controlling traffic signals on public roads.**
> Everything below describes what the code in this repository does and what has actually been measured
> (in simulation). Real-world deployment would require functional-safety certification, a conflict monitor
> unit in hardware, field validation and local regulatory approval.

A single-intersection controller that

1. detects vehicles / pedestrians / cyclists with YOLOv8 (PyTorch or ONNX Runtime),
2. tracks them (SORT: Kalman filter + Hungarian IoU matching),
3. estimates per-lane queues, arrivals, departures and waiting time and per-crosswalk pedestrian state,
4. lets a policy (fixed-time, actuated, or a DQN agent) **request** a signal phase, and
5. passes every request through an **independent safety manager** that is the only component allowed to
   produce signal lights.

## Architecture

```
frame ─► perception ─► tracking ─► state estimation ─► IntersectionObservation
          (detect)     (SORT)     (lanes, crosswalks,        │
                                   emergency confirm)        ▼
                                                     policy (fixed | actuated | DQN)
                                                     wrapped in GuardedPolicy (fallback)
                                                             │  PhaseRequest  (only a *request*)
                                                             ▼
                                         SignalSafetyManager  ──►  SignalCommand ─► signal output / SUMO
                                          (validates, runs the                      │
                                           yellow / all-red state machine)           ▼
                                                                              SignalAuditor (independent checker)
```

| Package | Responsibility |
|---|---|
| `src/perception/` | YOLOv8 detector (lazy load, CPU/CUDA, validated frames), shared post-processing, resilient `VideoSource` |
| `src/tracking/` | SORT tracker; stable ids, occlusion coasting, bounded memory, camera-gap handling |
| `src/queue_estimation/` | Lane polygons + stop lines, stopped-vs-moving, queue/arrival/departure/occupancy/wait per lane |
| `src/safety/` | `IntersectionModel` (conflict relation), **`SignalSafetyManager`**, **`SignalAuditor`**, `CrosswalkMonitor` |
| `src/emergency/` | Confirmation of emergency vehicles (multi-frame, confidence, hold); preemption itself is done by the safety manager |
| `src/control/` | Policies: `FixedTimePolicy`, `ActuatedPolicy`, `RLPolicy`, `GuardedPolicy` |
| `src/rl_agent/` | Observation encoder, reward, Gymnasium-style env, Double-DQN agent |
| `src/simulation/` | `SumoBackend` (real SUMO via TraCI), `QueueModelBackend` (fast stochastic model), sensor-noise model, scenario/demand generator |
| `src/edge/` | ONNX Runtime YOLOv8 detector, provider selection, benchmark |
| `src/controller.py` | Wires the stages together for camera operation |
| `src/state.py`, `src/errors.py` | Data structures between stages; exception types |

### Safety design

Safety is a hard constraint enforced *outside* the learning code (`src/safety/signal_safety.py`, no dependency on `rl_agent`).
The RL agent only emits a phase index; invalid or unsafe requests are rejected or delayed, and the reward function is **not**
relied on for safety.

Enforced by `SignalSafetyManager`:

* Phases come from an `IntersectionModel`; its constructor **rejects** any phase that serves two conflicting movements
  (approach vs approach, crosswalk vs approach). Default plan: `NS`, `EW` and an exclusive `PED` phase.
* Every change goes *hold → clearance (yellow, or flashing don't-walk for pedestrians) → all-red → next phase*. Greens are never swapped directly.
* Minimum green / walk; maximum green / walk (forces rotation, so a policy cannot starve a movement).
* A pedestrian walk runs at least `ped_walk_min_s`, is **extended** (up to `ped_walk_max_s`) while pedestrians are sensed in a crosswalk,
  is never cut short by vehicle demand, and always ends with a full flashing-don't-walk clearance.
* A new vehicle green is not released while a conflicting crosswalk is sensed occupied or vehicles are in the junction box;
  after `max_conflict_hold_s` the pedestrians are *served* rather than the vehicles released.
* Emergency preemption goes through the same state machine: it may skip the remaining minimum green of a vehicle phase, never clearance/all-red/minimum walk.
* Degraded sensing (`sensors_ok=False`: camera outage, repeated detector failure, stale data) → policy requests are ignored and a fixed cycle that
  serves **every** phase (including pedestrians) runs. Uncertain pedestrian detection adds all-red margin.
* Pedestrian protection does not rely on detection being complete: pedestrians only walk in an exclusive phase and every walk is followed by a full
  clearance and all-red interval, so a pedestrian the camera missed is still protected by timing.
* `SignalAuditor` re-checks the *displayed* lights (and, in SUMO runs, compares SUMO's actual TLS state with the commanded one) for conflicting greens,
  missing clearance, short yellow/all-red and short green.

Known safety limitations are listed under [Limitations](#known-limitations-and-unverified-assumptions).

## Verified features

"Verified" = covered by an automated test that passed in this environment (see [Tests](#tests)) and/or measured in SUMO.

| Feature | Status |
|---|---|
| Conflict-free phase model, yellow/all-red state machine, min/max timing, pedestrian walk protection, emergency preemption, degraded-sensor fallback | tested incl. property-based tests (random request/sensor sequences → 0 audit violations) |
| RL actions cannot bypass the safety layer (adversarial/invalid actions, sensor dropout) | tested in the queue model **and** SUMO |
| YOLOv8 PyTorch + ONNX Runtime paths, class filtering, CPU fallback, missing-weights error | tested with doubles; ONNX output cross-checked against Ultralytics on one real image (see Benchmarks) |
| SORT tracking: stable ids, occlusion coasting, no double counting, bounded memory | tested |
| Queue / stop-line departures / waiting time from tracks | tested on synthetic tracks only — **not validated on real video** |
| DQN training, checkpointing, seeded reproducibility, train/eval separation | tested; trained and evaluated in SUMO |
| SUMO integration: deterministic demand, reset/step/close cleanup, crash handling | tested against SUMO 1.28 |
| Jetson / Coral | **not implemented/verified** (see Edge deployment) |

## Installation

Python ≥ 3.10. Install only what you need:

```bash
pip install -r requirements.txt          # core: controller logic, safety layer, tracking, queue estimation
pip install -r requirements-vision.txt   # + OpenCV, ONNX Runtime (CPU), Ultralytics
pip install -r requirements-rl.txt       # + PyTorch, Gymnasium (DQN training)
pip install -r requirements-sim.txt      # + SUMO (pip eclipse-sumo), TraCI, sumolib
pip install -r requirements-dev.txt      # everything + pytest, hypothesis, ruff, mypy
```

* **Without SUMO**: skip `requirements-sim.txt`. The SUMO tests are skipped automatically; the `queue` backend still lets you train/test (`--backend queue`).
* **SUMO**: either `pip install eclipse-sumo` (found automatically) or install SUMO natively and set `SUMO_HOME`.
  The intersection network is generated with `netconvert` on first use; no network file is checked in.
* **GPU inference**: install `onnxruntime-gpu` *instead of* `onnxruntime` (matching CUDA/cuDNN required) and select `CUDAExecutionProvider`;
  for PyTorch, install a CUDA build of torch and set `perception.device: cuda`. Unavailable GPUs fall back to CPU with a warning.

## Model setup

No model weights are committed. Place files here (paths configurable in `config/default_config.yaml`):

| File | Purpose |
|---|---|
| `models/yolov8n.pt` | Ultralytics weights (`perception.model_path`). Weights are **never downloaded implicitly**; set `perception.allow_download: true` to let Ultralytics fetch stock weights by name. |
| `models/yolov8n.onnx` | ONNX export for `perception.backend: onnx` (`python -c "from src.edge.edge_inference import export_yolo_to_onnx as e; e('models/yolov8n.pt')"`) |
| `models/dqn_best.pt` | trained RL checkpoint (`python scripts/train.py …` writes `best.pt`/`last.pt`) |

Stock COCO weights detect person / bicycle / car / motorcycle / bus / truck. **Emergency vehicles can only be detected with custom weights
that contain classes such as `ambulance`, `fire_truck`, `police_car`** — there is no colour/flashing-light heuristic (it produced false
positives and was removed). Without such weights emergency preemption never triggers from vision.

## Configuration

`config/default_config.yaml` is the single source of defaults and every key is validated on load (`ConfigError` for unknown keys, wrong types, out-of-range values).
Override with a YAML file (`--config my_site.yaml`) and/or environment variables using double underscores for nesting:

```bash
TRAFFIC__PERCEPTION__DEVICE=cuda TRAFFIC__SIGNAL__TIMING__YELLOW_S=4.5 python scripts/run_controller.py --video v.mp4
```

Camera geometry is configuration, not code. Example `my_site.yaml`:

```yaml
queue_estimation:
  lanes:
    - {id: 0, approach: N, polygon: [[100,0],[200,0],[200,450],[100,450]], stop_line: [[100,400],[200,400]]}
    - {id: 1, approach: E, polygon: [[500,0],[600,0],[600,450],[500,450]], stop_line: [[500,400],[600,400]]}
crosswalks:
  - {id: cw_N, polygon: [[80,420],[220,420],[220,470],[80,470]]}
signal:
  policy: actuated        # fixed | actuated | rl
  timing: {yellow_s: 4.0, all_red_s: 2.0, min_green_s: 8.0, ped_clearance_s: 8.0}
```

Any number of lanes/approaches is supported; a different intersection geometry is described with `signal.intersection`
(approaches, crosswalks, phases, conflict pairs — validated by `IntersectionModel`). `ped_clearance_s` must be ≥ crossing width / walking speed for your site.

## Commands

```bash
# tests
python -m pytest                         # everything; SUMO tests skip if SUMO/traci is missing

# train (SUMO)  / train (fast queue model, no SUMO)
python scripts/train.py --backend sumo  --episodes 200 --seed 0 --out runs/dqn_sumo
python scripts/train.py --backend queue --episodes 50  --out runs/dqn_queue

# evaluate fixed-time vs actuated vs DQN on identical SUMO demand
python scripts/evaluate.py --checkpoint runs/dqn_sumo/best.pt --seeds 9000 9001 9002 9003 9004 --out runs/eval
python scripts/evaluate.py --checkpoint runs/dqn_sumo/best.pt --scenarios medium high --ped-detect-prob 0.6 --veh-detect-prob 0.8 --out runs/eval_noisy

# inference benchmark (+ torch/ONNX cross-check)
python scripts/benchmark_inference.py --onnx models/yolov8n.onnx --weights models/yolov8n.pt --image some.jpg

# run on video / camera (logs commands; does not drive hardware)
python scripts/run_controller.py --video traffic.mp4 --config my_site.yaml --log runs/commands.jsonl

# lint / types
ruff check src scripts tests && mypy
```

## SUMO scenarios and reproducibility

`src/simulation/sumo_scenario.py` builds a 4-arm, single-lane-per-direction signalised intersection with sidewalks and four crossings, and generates demand
from `(scenario, seed)` only (own `random.Random`, no global state; SUMO also receives `--seed`; teleporting is disabled so gridlock shows up in metrics).
Scenarios: `low`, `medium`, `high`, `rush_hour` (time-varying), `asymmetric`, `ped_heavy`, `emergency` (≈12 emergency vehicles/h).
Seeds: training 1000+, checkpoint selection 5000–5002, **final evaluation 9000+** (disjoint). Repeating an episode with the same seed gives identical metrics
(tested). Runs on CPU in about 4–5 s per 1800 s episode.

Metrics (all measured from SUMO / the safety layer, none estimated): per-vehicle accumulated waiting time (mean and max, including vehicles still in the network at the end),
mean travel time, pedestrian waiting time (person-seconds below 0.1 m/s), mean halting queue, throughput (vehicles passing the stop line), phase switches,
emergency-vehicle delay (accumulated waiting of `emg_*` vehicles), collisions, `tls_state_mismatch_steps` (SUMO displays something other than commanded),
audit violations, and `ped_conflict_steps` (ground truth: a pedestrian on a crossing while a conflicting approach showed green/yellow).

## Measured results (SUMO 1.28, simulation only)

Setup: 1800 s episodes, 5 evaluation seeds (9000–9004), mean over seeds, identical demand for every controller. All controllers use the same `SignalSafetyManager`
(so the same clearance/pedestrian rules); baselines decide every 1 s, the DQN every 5 s. DQN: Double-DQN, 200 training episodes (≈13 min CPU) on the 7 scenarios, checkpoint chosen on seeds 5000–5002.
Raw data: `scripts/evaluate.py` writes `results.json` / `results.md` (std and per-seed values included there).

| scenario | controller | veh wait mean [s] | veh wait max [s] | ped wait mean [s] | queue | throughput [veh/30 min] | phase switches |
|---|---|---|---|---|---|---|---|
| low | fixed_time | 20.8 | 69.8 | 32.1 | 3.2 | 272.2 | 57.2 |
| low | actuated | 11.8 | 59.0 | 14.7 | 1.8 | 274.4 | 106.0 |
| low | dqn | 10.2 | 53.2 | 54.2 | 1.6 | 274.6 | 93.8 |
| low | dqn (ped weight 6) | 27.9 | 136.0 | 14.9 | 4.3 | 271.2 | 78.0 |
| medium | fixed_time | 32.6 | 151.2 | 30.1 | 8.1 | 441.8 | 56.8 |
| medium | actuated | 25.4 | 92.4 | 22.2 | 6.3 | 445.8 | 102.2 |
| medium | dqn | 17.1 | 80.0 | 48.9 | 4.2 | 447.6 | 110.2 |
| medium | dqn (ped weight 6) | 38.4 | 140.8 | 18.4 | 9.7 | 440.6 | 107.0 |
| high | fixed_time | 45.0 | 219.0 | 31.8 | 13.8 | 534.6 | 56.8 |
| high | actuated | 42.5 | 165.4 | 30.0 | 13.0 | 539.0 | 82.8 |
| high | dqn | 27.6 | 97.4 | 45.5 | 8.5 | 546.6 | 114.4 |
| high | dqn (ped weight 6) | 63.5 | 208.2 | 23.9 | 19.7 | 522.4 | 113.8 |
| rush_hour | fixed_time | 55.9 | 270.2 | 33.9 | 14.6 | 456.8 | 56.6 |
| rush_hour | actuated | 49.4 | 211.2 | 27.2 | 12.9 | 465.0 | 85.0 |
| rush_hour | dqn | 33.7 | 132.4 | 46.8 | 8.9 | 470.4 | 110.8 |
| rush_hour | dqn (ped weight 6) | 60.5 | 182.4 | 21.8 | 16.0 | 457.2 | 105.8 |
| asymmetric | fixed_time | 44.4 | 208.6 | 31.4 | 9.6 | 375.0 | 57.0 |
| asymmetric | actuated | 20.4 | 88.6 | 17.8 | 4.4 | 390.8 | 103.2 |
| asymmetric | dqn | 12.9 | 66.0 | 44.6 | 2.8 | 392.4 | 94.8 |
| asymmetric | dqn (ped weight 6) | 31.3 | 96.2 | 13.7 | 6.9 | 388.4 | 94.0 |
| ped_heavy | fixed_time | 32.7 | 150.0 | 32.2 | 7.7 | 421.8 | 54.4 |
| ped_heavy | actuated | 44.8 | 175.8 | 18.6 | 10.6 | 418.2 | 82.4 |
| ped_heavy | dqn | 25.4 | 92.4 | 32.5 | 6.0 | 424.2 | 108.8 |
| ped_heavy | dqn (ped weight 6) | 95.2 | 329.4 | 14.2 | 22.7 | 399.2 | 105.4 |
| emergency | fixed_time | 28.9 | 138.2 | 32.7 | 7.0 | 436.4 | 57.6 |
| emergency | actuated | 24.2 | 96.4 | 23.2 | 5.9 | 437.2 | 102.6 |
| emergency | dqn | 16.9 | 81.8 | 48.6 | 4.1 | 435.4 | 110.6 |
| emergency | dqn (ped weight 6) | 35.5 | 155.4 | 21.6 | 8.7 | 432.8 | 100.0 |

Observations (honest reading):

* **Safety**: across all runs above (≈ 170 episodes) `safety_violations`, `ped_conflict_steps`, `collisions` and `tls_state_mismatch_steps` were **0**.
* The default DQN (pedestrian reward weight 1.5) reduces mean vehicle wait versus the actuated baseline in all 7 scenarios (e.g. high: 42.5 → 27.6 s, asymmetric: 20.4 → 12.9 s) with equal throughput (differences ≤ ~1–2 %, within seed noise),
  **but pedestrians wait considerably longer** (≈45–55 s vs 15–30 s for actuated; in `low` 54 s vs 15 s). It is *not* a Pareto improvement. In `ped_heavy` it beats actuated on vehicle wait (25.4 vs 44.8 s) with pedestrian waits (32.5 s) similar to fixed-time and worse than actuated (18.6 s).
* Raising the pedestrian weight to 6.0 (`dqn (ped weight 6)`) brings pedestrian waits down to 14–24 s but vehicle waits become **worse than actuated** (e.g. high: 63.5 s, ped_heavy: 95 s). A single reward weight therefore only slides along a trade-off;
  it did not produce a policy that dominates the actuated baseline. Choose the operating point deliberately; neither checkpoint is claimed to be better overall.
* Fixed-time is worst for vehicles but gives moderate, predictable pedestrian service. Emergency-vehicle delay (emergency scenario, mean accumulated wait of ~6 vehicles/episode): fixed 3.8 s, actuated 1.8 s, DQN 2.0 s — preemption is done by the safety manager for all controllers, so differences are small and noisy.
* Only 5 seeds per cell; std values in `results.json` are not negligible, so small differences (e.g. throughput) should not be over-interpreted. One training seed only.

**Sensor uncertainty** (pedestrian detection probability 0.6, vehicle detection 0.8; medium and high scenarios; DQN trained on clean sensors):

| scenario | controller | veh wait mean [s] | veh wait max [s] | ped wait mean [s] | queue | throughput [veh/30 min] | phase switches |
|---|---|---|---|---|---|---|---|
| medium | fixed_time | 35.2 | 172.2 | 35.1 | 8.8 | 442.2 | 54.0 |
| medium | actuated | 34.8 | 133.2 | 30.8 | 8.7 | 443.0 | 87.2 |
| medium | dqn | 25.8 | 100.6 | 49.4 | 6.4 | 443.6 | 99.0 |
| high | fixed_time | 61.4 | 260.0 | 34.8 | 18.7 | 523.0 | 54.0 |
| high | actuated | 63.0 | 239.8 | 34.9 | 19.4 | 527.6 | 71.8 |
| high | dqn | 54.5 | 160.8 | 50.6 | 16.9 | 528.0 | 101.6 |

Under degraded sensing all controllers lose performance (the DQN still has the lowest vehicle wait, with the same pedestrian-wait penalty) and the safety counters stay at 0. Dropout-to-fallback behaviour (`sensors_ok=False`) is covered by unit tests rather than by a SUMO sweep.


## Edge deployment

* `src/edge/OnnxYoloDetector` runs YOLOv8 ONNX exports on ONNX Runtime with letterbox pre-processing and class-wise NMS. Requested execution providers are intersected with
  `onnxruntime.get_available_providers()`; the providers actually active are reported. CPU always works.
* TensorRT / CUDA are used only if the installed onnxruntime build exposes those providers.
* **Jetson and Google Coral are not supported/verified.** The removed presets were unverified claims. Jetson would need a JetPack-matched onnxruntime/TensorRT build and
  benchmarking on the device; Coral needs a full-integer-quantised TFLite model and a different decode path. Neither has been attempted.
* INT8/FP16 quantisation is not implemented.

### Inference benchmark (measured, this machine)

Intel i-series CPU (Windows 11, Python 3.13), YOLOv8n, 640 input, 1080x810 frame, 50 timed runs after 10 warm-up runs, measured **while a training job was using the CPU**, so treat as pessimistic:

| runtime | active provider | mean latency | p95 | FPS | RSS |
|---|---|---|---|---|---|
| ONNX Runtime (incl. letterbox + decode + NMS) | CPUExecutionProvider | 101 ms | 145 ms | 9.9 | 135 MB |
| Ultralytics / PyTorch CPU | cpu | 149 ms | 237 ms | 6.7 | n/a |

Cross-check on Ultralytics' `bus.jpg`: PyTorch found 4 objects (bus + 3 persons), ONNX found the same 4 plus one extra borderline person; matched boxes differ by up to 19 px (letterbox/rounding differences). No GPU was available, so no CUDA numbers exist.


## Known limitations and unverified assumptions

* **Everything RL/traffic-related was measured in simulation only** (one synthetic intersection, one lane per direction, SUMO car-following). No real traffic, no real video was used.
  Queue/stop-line/waiting estimation from video is unit-tested on synthetic tracks only; accuracy on real footage is unknown.
* **Single un-calibrated camera**: pixel distances are not proportional to road distance (perspective). Queue length is therefore a vehicle *count*;
  metre values appear only if `meters_per_pixel` is set and are then approximate. "Stopped" is judged relative to box size, which is robust to scale but not to heavy occlusion
  (occluded vehicles are under-counted; coasting tracks mitigate short gaps only).
* Pedestrian detection can miss people (night, occlusion, children, wheelchairs). The design bounds the consequence with exclusive pedestrian timing and clearance, but there is no
  independent presence sensor, and the default exclusive-pedestrian plan is conservative (costs vehicle capacity). Concurrent walk with parallel traffic is not modelled.
* No protected left-turn phases or turn pockets: permitted lefts yield per SUMO right-of-way; the conflict model is approach-level only.
* Emergency preemption requires custom-trained emergency classes; there is no siren/transponder/V2X input. Priority is by approach, not by estimated arrival time.
* The safety manager is software. A real installation needs an independent hardware conflict monitor and fail-safe flash/all-red behaviour on controller failure.
* `ped_clearance_s` etc. are defaults, not engineering values; they must be derived for each site and standard.
* The DQN is trained on this one simulated geometry and these demand patterns; generalisation to other intersections or real demand is untested. PPO was deliberately not added.
* The baseline "actuated" controller is a simple gap-out/max-out policy written for this repo (parameters tuned on the queue model with separate seeds), not a commercial controller.

## Recommended next steps

1. Validate perception on labelled real footage from the target camera (queue counts, pedestrian recall by lighting condition) and add a homography calibration option.
2. Add protected-left phases and concurrent pedestrian phases to `IntersectionModel`, with the matching conflict data.
3. Compare against SUMO's built-in actuated/Webster controllers and multiple seeds/networks; report confidence intervals over more seeds.
4. Train with sensor noise and multiple geometries; consider constrained RL only after the safety layer has been reviewed by traffic engineers.
5. Hardware-in-the-loop test with a real controller + conflict monitor; Jetson/TensorRT benchmark on device.

## License / disclaimer

Provided as-is for research and education, without warranty of any kind. Do not use to control signals on public roads.
