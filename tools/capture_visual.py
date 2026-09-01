#!/usr/bin/env python3
"""Headless visual proof: run the pipeline on the live camera and save
annotated PNGs (no display required).

Each saved frame shows:
  - keypoints as circles colored by track ID (radius grows with track age)
  - green lines linking keypoints whose 64-bit code matched (Hamming <=
    threshold) between consecutive frames
  - a stats banner: fps, keypoints, matched pairs, mean matched Hamming

Usage:
    python3 tools/capture_visual.py [--seconds 15] [--every 45] \
        [--out_dir media/captures] [--max_keypoints 64] [--match_thresh 12]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from core.pipeline import FeaturePipeline  # noqa: E402
from core.projection import hamming64  # noqa: E402


def color_for(track_id: int):
    """Deterministic BGR color per track ID (HSV hue wheel)."""
    hue = (track_id * 47) % 180
    c = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive",
                        default=str(EXAMPLE_ROOT / "depthai_models" / "descriptor64_strip32_slim.tar.xz"))
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--every", type=int, default=45, help="save every Nth frame")
    parser.add_argument("--out_dir", type=Path, default=EXAMPLE_ROOT / "media" / "captures")
    parser.add_argument("--max_keypoints", type=int, default=64)
    parser.add_argument("--match_thresh", type=int, default=12)
    parser.add_argument("--tracker_decimate", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prev = {}  # track_id -> (x, y, code)
    saved = 0
    with FeaturePipeline(
        args.archive,
        max_keypoints=args.max_keypoints,
        fps_limit=30,
        publish_full_desc=False,
        nn_nodes=2,
        tracker_decimate=args.tracker_decimate,
    ) as fp:
        time.sleep(2)  # let tracks establish
        t0 = time.time()
        frames = 0
        stats = (0, 0.0)  # matched pairs, hamming sum
        while time.time() - t0 < args.seconds:
            res = fp.next_frame(timeout=5.0)
            if res is None:
                continue
            frame, kps, codes, _ = res
            frames += 1

            cur = {kp.track_id: (kp.x, kp.y, c) for kp, c in zip(kps, codes)}
            matches = []
            hsum, hcnt = 0.0, 0
            for tid, (x, y, c) in cur.items():
                if tid in prev:
                    d = int(hamming64(np.uint64(prev[tid][2]), np.uint64(c)))
                    if d <= args.match_thresh:
                        matches.append((tid, prev[tid][:2], (x, y), d))
                        hsum += d
                        hcnt += 1
            if hcnt:
                stats = (stats[0] + hcnt, stats[1] + hsum)

            if frames % args.every == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
                for tid, (px, py), (x, y), d in matches:
                    cv2.line(img, (int(px), int(py)), (int(x), int(y)),
                             (0, 220, 0), 1, cv2.LINE_AA)
                for kp in kps:
                    r = 2 + min(kp.age // 10, 4)
                    cv2.circle(img, (int(kp.x), int(kp.y)), r, color_for(kp.track_id),
                               1, cv2.LINE_AA)
                elapsed = time.time() - t0
                banner = (f"fps={frames/elapsed:.1f} kps={len(kps)} "
                          f"matched={hcnt} hamming_mean={hsum/max(hcnt,1):.1f}/64")
                cv2.putText(img, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
                out = args.out_dir / f"features_{frames:05d}.png"
                cv2.imwrite(str(out), img)
                saved += 1
                print(f"saved {out} ({banner})", flush=True)
            prev = cur

        dt = time.time() - t0
        print(f"RESULT: {frames/dt:.1f} fps, saved {saved} frames to {args.out_dir}, "
              f"matched-pair Hamming mean={stats[1]/max(stats[0],1):.1f}/64 over "
              f"{stats[0]} pairs", flush=True)


if __name__ == "__main__":
    main()
