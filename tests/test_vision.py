"""Perception, tracking, queue estimation, emergency confirmation, edge inference."""

import numpy as np
import pytest

from src.edge.edge_inference import OnnxYoloDetector, letterbox, nms, select_providers
from src.emergency import EmergencyVehicleHandler
from src.errors import DetectionError, ModelNotFoundError
from src.perception.object_detector import ObjectDetector, resolve_device
from src.perception.postprocess import FilterConfig, build_detections, validate_frame
from src.perception.video_source import VideoSource
from src.queue_estimation import QueueEstimator
from src.tracking import MultiObjectTracker
from tests.conftest import det_result, detection

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "ambulance", 4: "dog"}


# ------------------------------------------------------------------- detection
def test_build_detections_filters_invalid_and_irrelevant():
    xyxy = np.array([[10, 10, 110, 110], [10, 10, 12, 12], [np.nan, 0, 5, 5], [0, 0, 100, 100], [5, 5, 80, 80], [20, 20, 90, 90]])
    conf = np.array([0.9, 0.9, 0.9, 0.3, 0.9, 0.9])
    cls = np.array([2, 2, 2, 2, 4, 0])  # car, tiny car, nan, low-conf car, dog, person
    dets = build_detections(xyxy, conf, cls, NAMES, FilterConfig(0.5, min_box_area_px=100), (480, 640))
    assert [d.class_name for d in dets] == ["car", "person"]
    assert dets[0].is_vehicle and dets[1].is_pedestrian


def test_emergency_requires_model_class_and_high_confidence():
    cfg = FilterConfig(0.5, emergency_confidence=0.8)
    mk = lambda c, k: build_detections(np.array([[0, 0, 100, 100]]), np.array([c]), np.array([k]), NAMES, cfg, (480, 640))[0]
    assert mk(0.9, 3).is_emergency
    assert not mk(0.7, 3).is_emergency  # below the emergency bar: still a vehicle, not an emergency
    assert mk(0.7, 3).is_vehicle
    assert not mk(0.99, 2).is_emergency  # a red/blue ordinary car is never "emergency"


def test_per_class_threshold_and_box_fraction():
    cfg = FilterConfig(0.5, per_class_confidence={"person": 0.8})
    out = build_detections(np.array([[0, 0, 50, 50]] * 2), np.array([0.7, 0.7]), np.array([0, 2]), NAMES, cfg, (480, 640))
    assert [d.class_name for d in out] == ["car"]
    full = build_detections(np.array([[0, 0, 640, 480]]), np.array([0.9]), np.array([2]), NAMES, cfg, (480, 640))
    assert full == []


@pytest.mark.parametrize("bad", [None, np.zeros((10, 10)), np.zeros((4, 4, 3)), "x"])
def test_validate_frame_rejects_bad_frames(bad):
    with pytest.raises(DetectionError):
        validate_frame(bad)


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.array(xyxy, float), np.array(conf, float), np.array(cls)

    def __len__(self):
        return len(self.conf)


class FakeYolo:
    names = NAMES

    def __init__(self, boxes=None, fail=False):
        self.boxes, self.fail, self.calls = boxes, fail, 0

    def __call__(self, frame, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("CUDA OOM")

        class R:
            pass
        r = R()
        r.boxes = self.boxes
        return [r]


def test_detector_uses_injected_model_lazily_and_maps_classes():
    boxes = FakeBoxes([[10, 10, 200, 150], [300, 100, 380, 300]], [0.95, 0.9], [3, 0])
    d = ObjectDetector(model=FakeYolo(boxes))
    res = d.detect(np.zeros((480, 640, 3), np.uint8), 5.0)
    assert res.timestamp == 5.0 and res.emergency_count == 1 and res.pedestrian_count == 1


def test_detector_wraps_model_failure_and_rejects_missing_frame():
    d = ObjectDetector(model=FakeYolo(fail=True))
    with pytest.raises(DetectionError, match="inference failed"):
        d.detect(np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(DetectionError):
        d.detect(None)


def test_missing_weights_raise_instead_of_downloading(tmp_path):
    d = ObjectDetector(model_path=str(tmp_path / "nope.pt"))
    with pytest.raises(ModelNotFoundError):
        d.detect(np.zeros((480, 640, 3), np.uint8))


def test_cuda_request_falls_back_to_cpu_without_cuda(monkeypatch):
    import sys
    import types
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_video_source_survives_failed_reads_and_reconnects():
    class Cap:
        opened = 0

        def __init__(self, src):
            Cap.opened += 1
            self.n = 0
            self.first = Cap.opened == 1

        def isOpened(self):
            return True

        def read(self):
            self.n += 1
            return (False, None) if self.first and self.n <= 6 else (True, np.zeros((20, 20, 3), np.uint8))

        def release(self):
            pass

    vs = VideoSource(0, reconnect_after=3, capture_factory=Cap)
    results = [vs.read()[0] for _ in range(4)]
    assert [r is None for r in results] == [True, True, True, False]  # recovered after reconnect
    assert vs.total_failures == 3 and vs.reconnects == 1 and vs.consecutive_failures == 0


# ---------------------------------------------------------------------- tracking
def moving_box(i, x0=100, dx=4):
    return detection(x0 + dx * i, 100, x0 + dx * i + 60, 140)


def test_tracker_keeps_one_id_for_a_moving_vehicle():
    tr = MultiObjectTracker(min_hits=1)
    ids = set()
    for i in range(40):
        for t in tr.update(det_result(i * 0.1, [moving_box(i)])):
            ids.add(t.track_id)
    assert len(ids) == 1


def test_tracker_ids_are_unique_for_simultaneous_objects_and_not_double_counted():
    tr = MultiObjectTracker(min_hits=1)
    for i in range(30):
        tr.update(det_result(i * 0.1, [moving_box(i), detection(400, 300 + i, 460, 340 + i)]))
    assert tr.get_statistics()["total_tracks_created"] == 2


def test_tracker_survives_occlusion_with_same_id_and_reports_coasting():
    tr = MultiObjectTracker(max_age=10, min_hits=1, output_coasting=5)
    ids, coasting = set(), 0
    for i in range(40):
        dets = [] if 15 <= i < 20 else [moving_box(i)]
        for t in tr.update(det_result(i * 0.1, dets)):
            ids.add(t.track_id)
            coasting += t.is_coasting
    assert len(ids) == 1 and coasting >= 3


def test_tracker_forgets_objects_and_bounds_memory():
    tr = MultiObjectTracker(max_age=3, min_hits=1)
    for i in range(300):
        tr.update(det_result(i * 0.1, [detection(10 * (i % 50), 10, 10 * (i % 50) + 40, 60)] if i % 7 == 0 else []))
    assert len(tr.tracks) <= len(tr.trackers) <= 3
    tr2 = MultiObjectTracker(max_age=3, min_hits=1)
    for i in range(20):
        tr2.update(det_result(i * 0.1, [detection(i * 200 % 500, 10, i * 200 % 500 + 40, 60)]))
    for i in range(20, 40):
        tr2.update(det_result(i * 0.1, []))
    assert tr2.tracks == {}


def test_tracker_drops_invalid_detections():
    tr = MultiObjectTracker(min_hits=1)
    tr.update(det_result(0, [detection(10, 10, 5, 5), detection(np.nan, 0, 10, 10), detection(0, 0, 50, 50)]))
    assert tr.get_statistics()["dropped_invalid_detections"] == 2 and len(tr.tracks) == 1


def test_camera_gap_drops_tracks_and_ids_stay_unique():
    tr = MultiObjectTracker(min_hits=1, max_gap_s=1.0)
    first = {t.track_id for i in range(5) for t in tr.update(det_result(i * 0.1, [moving_box(i)]))}
    after = {t.track_id for i in range(5) for t in tr.update(det_result(30 + i * 0.1, [moving_box(i + 5)]))}
    assert first.isdisjoint(after)


# ------------------------------------------------------------------------- queue
def lane_estimator():
    est = QueueEstimator(min_stop_s=1.0)
    # approach "N": vertical lane x in [100,200], vehicles travel downwards; stop line at y=400 (directed left->right)
    est.add_lane(0, "north", polygon=[[100, 0], [200, 0], [200, 450], [100, 450]],
                 stop_line=[(100, 400), (200, 400)], approach="N")
    return est


def feed(est, tr, frames):
    states = None
    for ts, dets in frames:
        states = est.update(tr.update(det_result(ts, dets)), ts)
    return states


def test_stopped_vehicles_form_queue_and_moving_ones_do_not():
    est, tr = lane_estimator(), MultiObjectTracker(min_hits=1)
    frames = []
    for i in range(30):
        ts = i * 0.2
        frames.append((ts, [detection(120, 330, 180, 390),  # stopped at the stop line
                            detection(120, 240, 180, 290),  # stopped behind it
                            detection(130, 20 + i * 8, 190, 80 + i * 8)]))  # moving
    st = feed(est, tr, frames)[0]
    assert st.vehicle_count == 3 and st.queue_length == 2
    assert st.max_waiting_time >= 4.0
    assert est.approach_states()["N"]["queue"] == 2


def test_stop_line_crossing_counted_once_and_rates_use_frame_time():
    est, tr = lane_estimator(), MultiObjectTracker(min_hits=1)
    frames = [(i * 0.2, [detection(120, 200 + i * 15, 180, 260 + i * 15)]) for i in range(40)]
    feed(est, tr, frames)
    lane = est.lanes[0]
    assert len(lane._departures) == 1
    assert lane.state.departure_rate == pytest.approx(60 / 60.0)  # 1 vehicle in the 60 s window
    assert lane.state.arrival_rate == pytest.approx(1.0)


def test_vehicle_outside_any_lane_is_ignored_and_pedestrians_not_counted():
    est, tr = lane_estimator(), MultiObjectTracker(min_hits=1)
    st = feed(est, tr, [(0.0, [detection(500, 100, 560, 160), detection(120, 100, 150, 180, cls="person")])])[0]
    assert st.vehicle_count == 0


def test_lane_config_validation():
    est = QueueEstimator()
    with pytest.raises(ValueError):
        est.add_lane(0, "n", polygon=[[0, 0], [1, 1]])  # < 3 points
    est.add_lane(1, "n", polygon=[[0, 0], [10, 0], [10, 10]])
    with pytest.raises(ValueError):
        est.add_lane(1, "n", polygon=[[0, 0], [10, 0], [10, 10]])  # duplicate id


def test_queue_estimator_from_config_supports_arbitrary_lane_count():
    cfg = {"lanes": [{"id": i, "approach": "A", "polygon": [[i * 60, 0], [i * 60 + 50, 0], [i * 60 + 50, 300], [i * 60, 300]]}
                     for i in range(5)]}
    assert len(QueueEstimator.from_config(cfg).lanes) == 5


# --------------------------------------------------------------------- emergency
def emerg_track(tid, conf=0.9, coasting=False):
    from src.tracking.track import Track
    return Track(tid, "ambulance", 3, bbox=np.array([0, 0, 50, 50.0]), confidence=conf, is_vehicle=True,
                 is_emergency=True, is_coasting=coasting)


def test_emergency_needs_repeated_confident_sightings_and_holds_after_loss():
    h = EmergencyVehicleHandler(0.7, 3, hold_s=4.0, approach_of=lambda t: "E")
    assert h.update([emerg_track(1)], 0.0) == frozenset()
    assert h.update([emerg_track(1)], 0.1) == frozenset()
    assert h.update([emerg_track(1)], 0.2) == {"E"}
    assert h.update([], 3.0) == {"E"}  # held through occlusion
    assert h.update([], 5.0) == frozenset()


def test_low_confidence_or_coasting_emergency_does_not_confirm():
    h = EmergencyVehicleHandler(0.7, 2, approach_of=lambda t: "N")
    for i in range(5):
        assert h.update([emerg_track(1, conf=0.5), emerg_track(2, coasting=True)], i * 0.1) == frozenset()


# ------------------------------------------------------------------------- edge
def test_provider_selection_skips_unavailable_and_keeps_cpu():
    assert select_providers(("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
                            available=["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert select_providers(("CUDAExecutionProvider",), available=["CUDAExecutionProvider", "CPUExecutionProvider"]) \
        == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_letterbox_and_nms():
    img = np.full((100, 200, 3), 7, np.uint8)
    out, scale, (px, py) = letterbox(img, 64)
    assert out.shape == (64, 64, 3) and scale == pytest.approx(0.32) and py > 0 and px == 0
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]


class FakeSession:
    def __init__(self, out):
        self.out = out

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        assert feed["images"].shape == (1, 3, 64, 64) and feed["images"].dtype == np.float32
        return [self.out]


def test_onnx_detector_decodes_yolov8_layout_and_maps_back_to_frame():
    # [1, 4+C, N]: one confident car centred at (32,32) size 20x20 in 64x64 letterboxed space
    n_cls = 80
    pred = np.zeros((1, 4 + n_cls, 3), np.float32)
    pred[0, :4, 0] = [32, 32, 20, 20]
    pred[0, 4 + 2, 0] = 0.9  # class 2 = car
    pred[0, :4, 1] = [33, 33, 20, 20]  # duplicate -> removed by NMS
    pred[0, 4 + 2, 1] = 0.8
    det = OnnxYoloDetector("unused.onnx", session=FakeSession(pred), input_size=64,
                           filter_config=FilterConfig(0.5, min_box_area_px=1))
    frame = np.zeros((128, 128, 3), np.uint8)  # scale 0.5, no padding
    res = det.detect(frame, 3.0)
    assert len(res.detections) == 1 and res.detections[0].class_name == "car"
    b = res.detections[0].bbox
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((44, 44, 84, 84))
    assert det.active_providers == ["CPUExecutionProvider"]


def test_onnx_detector_rejects_unexpected_output_and_missing_model(tmp_path):
    det = OnnxYoloDetector("unused.onnx", session=FakeSession(np.zeros((2, 5))), input_size=64)
    with pytest.raises(DetectionError):
        det.detect(np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(ModelNotFoundError):
        OnnxYoloDetector(str(tmp_path / "missing.onnx"))
