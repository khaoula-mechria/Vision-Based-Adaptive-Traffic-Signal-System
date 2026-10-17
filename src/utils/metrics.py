"""
Metrics collection and reporting
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric measurement"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and reports performance metrics.

    Tracks:
    - Average and maximum waiting times (vehicles & pedestrians)
    - Queue length per lane
    - Intersection throughput
    - Emergency vehicle clearance time
    - System latency and inference speed
    """

    def __init__(
        self,
        collection_interval: float = 5.0,
        history_size: int = 1000,
    ):
        """
        Initialize metrics collector.

        Args:
            collection_interval: Minimum interval between collections
            history_size: Maximum number of data points to keep
        """
        self.collection_interval = collection_interval
        self.history_size = history_size

        # Metric storage
        self._metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._last_collection: float = 0

        # Aggregation
        self._running_totals: Dict[str, float] = defaultdict(float)
        self._running_counts: Dict[str, int] = defaultdict(int)

        # Start time for uptime tracking
        self._start_time = time.time()

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
        """
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )

        self._metrics[name].append(metric)

        # Update running statistics
        self._running_totals[name] += value
        self._running_counts[name] += 1

        # Trim history
        if len(self._metrics[name]) > self.history_size:
            self._metrics[name] = self._metrics[name][-self.history_size:]

    def record_waiting_time(
        self,
        waiting_time: float,
        entity_type: str = "vehicle",
        lane_id: Optional[int] = None,
    ):
        """Record waiting time measurement"""
        tags = {"type": entity_type}
        if lane_id is not None:
            tags["lane"] = str(lane_id)

        self.record(f"{entity_type}_waiting_time", waiting_time, tags)

    def record_queue_length(
        self,
        queue_length: int,
        lane_id: int,
    ):
        """Record queue length for a lane"""
        self.record(
            "queue_length",
            float(queue_length),
            {"lane": str(lane_id)},
        )

    def record_throughput(
        self,
        vehicles_passed: int,
        period_seconds: float,
    ):
        """Record intersection throughput"""
        # Convert to vehicles per hour
        vph = (vehicles_passed / period_seconds) * 3600 if period_seconds > 0 else 0
        self.record("throughput_vph", vph)

    def record_emergency_clearance(
        self,
        clearance_time: float,
        vehicle_type: str = "unknown",
    ):
        """Record emergency vehicle clearance time"""
        self.record(
            "emergency_clearance_time",
            clearance_time,
            {"vehicle_type": vehicle_type},
        )

    def record_latency(
        self,
        latency_ms: float,
        component: str = "inference",
    ):
        """Record system latency"""
        self.record(
            f"{component}_latency_ms",
            latency_ms,
            {"component": component},
        )

    def get_average(self, name: str, window_seconds: Optional[float] = None) -> float:
        """
        Get average value for a metric.

        Args:
            name: Metric name
            window_seconds: Time window (None for all time)

        Returns:
            Average value
        """
        if name not in self._metrics:
            return 0.0

        values = self._get_values_in_window(name, window_seconds)

        if not values:
            return 0.0

        return sum(values) / len(values)

    def get_max(self, name: str, window_seconds: Optional[float] = None) -> float:
        """Get maximum value for a metric"""
        if name not in self._metrics:
            return 0.0

        values = self._get_values_in_window(name, window_seconds)

        if not values:
            return 0.0

        return max(values)

    def get_min(self, name: str, window_seconds: Optional[float] = None) -> float:
        """Get minimum value for a metric"""
        if name not in self._metrics:
            return 0.0

        values = self._get_values_in_window(name, window_seconds)

        if not values:
            return 0.0

        return min(values)

    def _get_values_in_window(
        self,
        name: str,
        window_seconds: Optional[float],
    ) -> List[float]:
        """Get metric values within time window"""
        metrics = self._metrics.get(name, [])

        if window_seconds is None:
            return [m.value for m in metrics]

        cutoff = time.time() - window_seconds
        return [m.value for m in metrics if m.timestamp > cutoff]

    def get_summary(self, window_seconds: float = 300.0) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Args:
            window_seconds: Time window for calculations

        Returns:
            Summary dictionary
        """
        summary = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "metrics": {},
        }

        for name in self._metrics.keys():
            values = self._get_values_in_window(name, window_seconds)
            if values:
                summary["metrics"][name] = {  # type: ignore[index]
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        # Add specific computed metrics
        summary["computed"] = {
            "avg_vehicle_waiting_time": self.get_average("vehicle_waiting_time", window_seconds),
            "max_vehicle_waiting_time": self.get_max("vehicle_waiting_time", window_seconds),
            "avg_pedestrian_waiting_time": self.get_average("pedestrian_waiting_time", window_seconds),
            "avg_queue_length": self.get_average("queue_length", window_seconds),
            "avg_throughput_vph": self.get_average("throughput_vph", window_seconds),
            "avg_inference_latency_ms": self.get_average("inference_latency_ms", window_seconds),
        }

        return summary

    def get_evaluation_metrics(self) -> Dict[str, float]:
        """
        Get metrics for evaluation comparison.

        Returns metrics defined in project requirements.
        """
        return {
            "average_vehicle_waiting_time": self.get_average("vehicle_waiting_time"),
            "max_vehicle_waiting_time": self.get_max("vehicle_waiting_time"),
            "average_pedestrian_waiting_time": self.get_average("pedestrian_waiting_time"),
            "max_pedestrian_waiting_time": self.get_max("pedestrian_waiting_time"),
            "average_queue_length": self.get_average("queue_length"),
            "total_throughput": self._running_totals.get("throughput_vph", 0) / max(1, self._running_counts.get("throughput_vph", 1)),
            "average_emergency_clearance_time": self.get_average("emergency_clearance_time"),
            "average_inference_latency_ms": self.get_average("inference_latency_ms"),
        }

    def export_to_json(self, path: str):
        """Export metrics to JSON file"""
        try:
            data = {
                "summary": self.get_summary(),
                "evaluation_metrics": self.get_evaluation_metrics(),
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Metrics exported to {path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def reset(self):
        """Reset all metrics"""
        self._metrics.clear()
        self._running_totals.clear()
        self._running_counts.clear()
        self._start_time = time.time()
