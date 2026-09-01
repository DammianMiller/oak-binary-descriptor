#!/usr/bin/env python3
"""Sweep the detection-only pipeline to find the maximum feature count that
still sustains a given frame rate, with the ThresholdAutotuner converging the
Harris threshold onto each target.

For each target (ascending), the pipeline is rebuilt with
numTargetFeatures=target, the tuner converges for --converge seconds, then
fps/feature count are measured over --measure seconds. The sweep stops at the
first target that drops below 97% of the requested frame rate.

Usage: python3 tools/sweep_autotune.py --fps 60 --targets 300 400 500 600 684
"""

import argparse
import sys
import time
from pathlib import Path

import depthai as dai

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from core.pipeline import ThresholdAutotuner  # noqa: E402


def run_target(fps_limit: float, target: int, converge_s: float, measure_s: float,
               tune: bool = True, initial_threshold: float = 200.0):
    """Returns (fps, avg_features, final_threshold) for one target.

    With tune=False the threshold stays at the initial 200 (control run that
    isolates the cost of the emitted-count cap from the cost of a low
    threshold's candidate load).
    """
    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            gray = cam.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=fps_limit)
            tracker = pipeline.create(dai.node.FeatureTracker)
            tracker.initialConfig.setCornerDetector(
                dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
            )
            tracker.initialConfig.setNumTargetFeatures(target)
            if initial_threshold != 200.0:
                cd = dai.FeatureTrackerConfig.CornerDetector()
                cd.type = dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
                cd.thresholds.initialValue = initial_threshold
                tracker.initialConfig.setCornerDetector(cd)
            tracker.initialConfig.setMotionEstimator(False)
            tracker.setHardwareResources(2, 2)
            gray.link(tracker.inputImage)
            qf = tracker.outputFeatures.createOutputQueue(maxSize=30, blocking=False)
            qi = tracker.passthroughInputImage.createOutputQueue(maxSize=30, blocking=False)
            qc = tracker.inputConfig.createInputQueue(maxSize=4, blocking=False)
            pipeline.start()
            tuner = ThresholdAutotuner(qc, target, initial=initial_threshold, high=8000.0)
            time.sleep(1)

            # Converge: drain both queues, feed the tuner.
            t_end = time.time() + converge_s
            while time.time() < t_end:
                km = qf.tryGet()
                if km is not None and tune:
                    tuner.update(len(km.trackedFeatures))
                qi.tryGet()
                time.sleep(0.001)

            # Measure.
            frames = 0
            counts = []
            t0 = time.time()
            while time.time() - t0 < measure_s:
                fm = qi.tryGet()
                if fm is not None:
                    frames += 1
                km = qf.tryGet()
                if km is not None:
                    counts.append(len(km.trackedFeatures))
                    if tune:
                        tuner.update(len(km.trackedFeatures))
                time.sleep(0.001)
            fps = frames / (time.time() - t0)
            avg = sum(counts) / len(counts) if counts else 0.0
            return fps, avg, tuner.threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, required=True, help="sensor frame rate limit")
    ap.add_argument("--targets", type=int, nargs="+", required=True)
    ap.add_argument("--converge", type=float, default=10.0)
    ap.add_argument("--measure", type=float, default=5.0)
    ap.add_argument("--no_tune", action="store_true",
                    help="control: keep the initial threshold, only cap the count")
    ap.add_argument("--threshold", type=float, default=200.0,
                    help="initial Harris threshold (used as the fixed value with --no_tune)")
    args = ap.parse_args()

    print(f"fps_limit={args.fps}  (converge {args.converge}s, measure {args.measure}s per target)")
    print(f"{'target':>6} {'fps':>7} {'features':>9} {'threshold':>10}  holds?")
    best = None
    for target in sorted(args.targets):
        try:
            fps, avg, thr = run_target(args.fps, target, args.converge, args.measure,
                                       tune=not args.no_tune,
                                       initial_threshold=args.threshold)
        except Exception as e:  # device crash / XLink error: record and stop
            print(f"{target:>6} {'CRASH':>7} {'':>9} {'':>10}  {type(e).__name__}: {e}")
            break
        holds = fps >= 0.97 * args.fps
        print(f"{target:>6} {fps:>7.1f} {avg:>9.1f} {thr:>10.0f}  {'yes' if holds else 'NO'}")
        if holds:
            best = (target, avg, fps)
        else:
            break
    if best:
        print(f"\nmax sustained at {args.fps:.0f} fps: target {best[0]} "
              f"({best[1]:.0f} features/frame @ {best[2]:.1f} fps)")
    else:
        print(f"\nno target sustained {args.fps:.0f} fps")


if __name__ == "__main__":
    main()
