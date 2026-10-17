#!/usr/bin/env python
"""Run the vision-based controller on a video file or camera and log the signal commands.

    python scripts/run_controller.py --video traffic.mp4 --config my_site.yaml --log runs/commands.jsonl

The camera layout (lane polygons, stop lines, crosswalk polygons) must be supplied in the config
(``queue_estimation.lanes`` / ``crosswalks``); see the README. This script only *logs* the commands;
it does not drive real signal hardware. Research prototype - not for public-road use.
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.controller import AdaptiveTrafficController  # noqa: E402
from src.perception.video_source import VideoSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file path")
    src.add_argument("--camera", type=int, help="camera index")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--log", default=None, help="write one JSON line per step")
    ap.add_argument("--status-every", type=int, default=100)
    args = ap.parse_args()

    ctl = AdaptiveTrafficController(config_path=args.config)
    logging.basicConfig(level=getattr(logging, ctl.config.get("system.log_level")),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    source = VideoSource(args.video if args.video else args.camera)
    log = open(args.log, "w", encoding="utf-8") if args.log else None
    n, t_start = 0, time.time()
    try:
        while args.max_frames is None or n < args.max_frames:
            frame, wall_ts = source.read()
            if frame is None and args.video:
                break  # end of file
            # file input: use media time so replay is deterministic; live camera: wall clock
            ts = n / 25.0 if args.video else wall_ts
            res = ctl.step(frame, ts)
            n += 1
            if log:
                log.write(json.dumps({"t": ts, "phase": res.command.phase, "stage": res.command.stage.value,
                                      "approach": res.command.approach_lights, "crosswalk": res.command.crosswalk_lights,
                                      "sensors_ok": res.sensors_ok, "events": res.command.events,
                                      "ms": round(res.processing_ms, 1)}) + "\n")
            if n % args.status_every == 0:
                logging.info("status: %s", ctl.get_status())
    finally:
        source.release()
        if log:
            log.close()
    logging.info("processed %d frames in %.1fs; audit violations: %d", n, time.time() - t_start, ctl.auditor.total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
