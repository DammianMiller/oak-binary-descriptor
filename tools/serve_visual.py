#!/usr/bin/env python3
"""Realtime MJPEG capture service: live annotated feature feed over HTTP.

Runs FeaturePipeline on the live camera in a background thread, annotates
each frame (keypoints colored by track ID, green match lines between
consecutive frames where the 64-bit codes agree within --match_thresh, stats
banner), and serves it as an MJPEG stream — viewable in any browser, no
display required.

Endpoints:
    /        -> simple HTML page embedding the stream
    /stream  -> multipart MJPEG stream

Usage:
    python3 tools/serve_visual.py [--port 8081] [--max_keypoints 64]
Then open http://<host>:8081/ (works over LAN too).
"""

import argparse
import collections
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(EXAMPLE_ROOT / "tools"))

from capture_visual import color_for  # noqa: E402
from core.pipeline import FeaturePipeline, ThresholdAutotuner  # noqa: E402
from core.projection import hamming64  # noqa: E402

HTML = b"""<!doctype html>
<html><head><title>OAK-1 features</title></head>
<body style="background:#111;color:#eee;font-family:monospace">
<h3>OAK-1 live features (640x400, 64-bit codes)</h3>
<img src="/stream" style="max-width:100%"/>
</body></html>"""


class CaptureService:
    """Background pipeline loop keeping the latest annotated JPEG."""

    def __init__(self, args):
        self._args = args
        self._jpeg = None
        self._lock = threading.Lock()
        self.stats = "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        # The pipeline thread needs up to ~1 s (next_frame timeout) plus the
        # graceful-teardown drain (~1 s) to exit its device context manager.
        self._thread.join(timeout=8)

    def latest_jpeg(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._jpeg is not None:
                    return self._jpeg
            time.sleep(0.01)
        return None

    def _run_extraction_only(self):
        """Detection-only fast path: Harris corners per frame, no tracking
        (motion estimator disabled), no descriptors. Measured flat ~60 fps at
        up to 1000 target features; the descriptor pipeline tops out at 30.
        """
        args = self._args
        frames = 0
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                cam = pipeline.create(dai.node.Camera).build()
                gray = cam.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8,
                                         fps=float(args.fps))
                tracker = pipeline.create(dai.node.FeatureTracker)
                tracker.initialConfig.setCornerDetector(
                    dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
                )
                # Aim the detector just under the measured ceiling for the
                # sensor rate: ~390 features at 60 fps (below threshold ~200
                # the detector overruns the 16 ms budget and collapses to
                # ~46 fps / a truncated ~320 grid) and ~750 at 30 fps (XLink
                # drops TrackedFeatures metadata over 51200 B). Build the
                # tracker with this cap too — a runtime-only
                # numTargetFeatures change does not resize the detection
                # grid, and a build-time cap above the tuner target lets the
                # count overshoot the target and walk the threshold to its
                # ceiling (measured).
                ceiling = 380 if args.fps > 45 else 700
                aim = min(args.max_keypoints, ceiling)
                tracker.initialConfig.setNumTargetFeatures(aim)
                tracker.initialConfig.setMotionEstimator(False)
                tracker.setHardwareResources(2, 2)
                gray.link(tracker.inputImage)
                # passthrough gives the aligned frame to draw on
                qf = tracker.outputFeatures.createOutputQueue(maxSize=30, blocking=False)
                qi = tracker.passthroughInputImage.createOutputQueue(maxSize=30, blocking=False)
                qc = tracker.inputConfig.createInputQueue(maxSize=4, blocking=False)
                pipeline.start()
                # Lighting autotune: converge the feature count on the aim by
                # nudging the Harris threshold; the fps guard ratchets the
                # floor back up if a threshold lowering ever trips the
                # detector-budget cliff.
                tuner = ThresholdAutotuner(qc, aim, min_fps=0.95 * args.fps)
                time.sleep(1)
                t0 = time.time()
                frame_times = collections.deque(maxlen=90)  # ~1.5 s rolling window
                while not self._stop.is_set():
                    fm = qi.tryGet()
                    if fm is None:
                        time.sleep(0.001)
                        continue
                    frame = fm.getCvFrame()
                    frame_times.append(time.time())
                    roll_fps = (
                        (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                        if len(frame_times) > 10 else None
                    )
                    feats = []
                    km = qf.tryGet()
                    if km is not None:
                        feats = list(km.trackedFeatures)
                        tuner.update(len(feats), fps=roll_fps)
                    frames += 1
                    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    for f in feats:
                        cv2.circle(img, (int(f.position.x), int(f.position.y)), 2,
                                   color_for(int(f.id) if int(f.id) >= 0 else 0),
                                   1, cv2.LINE_AA)
                    banner = (f"EXTRACTION-ONLY fps={frames/(time.time()-t0):.1f} "
                              f"features={len(feats)} threshold={tuner.threshold:.0f} "
                              f"(no tracking, no descriptors)")
                    cv2.putText(img, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._lock:
                            self._jpeg = jpeg.tobytes()
                        self.stats = banner

    def _run(self):
        if self._args.extraction_only:
            return self._run_extraction_only()
        args = self._args
        prev = {}
        frames = 0
        t0 = time.time()
        no_track = args.no_tracking
        gated = args.describe_gate
        with FeaturePipeline(
            args.archive,
            max_keypoints=args.max_keypoints,
            describe_budget=args.describe_budget,
            fps_limit=30,
            publish_full_desc=False,
            nn_nodes=2,
            tracker_decimate=args.tracker_decimate,
            tracking=not no_track,
            describe_gate=gated,
            change_thresh=args.change_thresh,
            refresh_every=args.refresh_every,
        ) as fp:
            time.sleep(1)
            t0 = time.time()
            while not self._stop.is_set():
                try:
                    # Short timeout so the loop notices the stop event
                    # quickly and the pipeline context manager gets to close
                    # the device gracefully (see the SIGTERM handler in
                    # main()). Kept small because _drain_job applies it per
                    # strip read — 1.0 s with an 8-strip config blocked
                    # shutdown for >15 s (measured).
                    res = fp.next_frame(timeout=0.3)
                except RuntimeError as e:
                    # Device died mid-session (loud error from the pipeline);
                    # leave the loop so the context manager exits. The dead
                    # session is deliberately NOT closed (depthai segfaults
                    # closing a crashed device).
                    self.stats = f"DEVICE LOST: {e}"
                    print(self.stats, flush=True)
                    break
                if res is None:
                    continue
                frame, kps, codes, _ = res
                frames += 1

                # Legacy/budget mode: codes align with the FIRST len(codes)
                # keypoints (oldest tracks). Gate mode: codes are parallel
                # and each record's has_code/code_age says valid/how stale.
                if gated:
                    n_described = sum(1 for kp in kps if kp.has_code)
                    n_fresh = sum(1 for kp in kps if kp.has_code and kp.code_age == 0)
                else:
                    n_described = len(codes)
                    n_fresh = n_described
                hsum, hcnt = 0.0, 0
                img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
                if no_track:
                    # Detection + description only: matching is disabled
                    # (consumers, e.g. ROS subscribers, do their own code
                    # matching). Dots only.
                    pass
                else:
                    cur = {kp.track_id: (kp.x, kp.y, c)
                           for kp, c in zip(kps, codes) if kp.has_code}
                    for tid, (x, y, c) in cur.items():
                        if tid in prev:
                            d = int(hamming64(np.uint64(prev[tid][2]), np.uint64(c)))
                            if d <= args.match_thresh:
                                px, py = prev[tid][:2]
                                cv2.line(img, (int(px), int(py)), (int(x), int(y)),
                                         (0, 220, 0), 1, cv2.LINE_AA)
                                hsum += d
                                hcnt += 1
                # Codeless keypoints first (dim), coded on top (gate mode
                # dims stale cached codes to half-bright, fresh stay full).
                if gated:
                    coded = [(i, kp) for i, kp in enumerate(kps) if kp.has_code]
                    uncoded = [kp for kp in kps if not kp.has_code]
                else:
                    coded = list(enumerate(kps[:n_described]))
                    uncoded = kps[n_described:]
                for kp in uncoded:
                    cv2.circle(img, (int(kp.x), int(kp.y)), 1,
                               (90, 90, 90), 1, cv2.LINE_AA)
                for i, kp in coded:
                    r = 2 + min(kp.age // 10, 4)
                    ident = kp.track_id if kp.track_id >= 0 else i
                    col = color_for(ident)
                    if gated and kp.code_age > 0:
                        col = tuple(int(v * 0.5) for v in col)
                    cv2.circle(img, (int(kp.x), int(kp.y)), r, col, 1, cv2.LINE_AA)
                mode = "NO-TRACK " if no_track else ""
                if no_track:
                    banner = (f"{mode}fps={frames/(time.time()-t0):.1f} "
                              f"kps={len(kps)} described={n_described}"
                              + (f"({n_fresh} fresh) " if gated else " ")
                              + "(matching disabled)")
                elif gated:
                    banner = (f"GATED fps={frames/(time.time()-t0):.1f} "
                              f"kps={len(kps)} described={n_described}"
                              f"({n_fresh} fresh) matched={hcnt} "
                              f"hamming_mean={hsum/max(hcnt,1):.1f}/64")
                else:
                    banner = (f"fps={frames/(time.time()-t0):.1f} kps={len(kps)} "
                              f"described={n_described} matched={hcnt} "
                              f"hamming_mean={hsum/max(hcnt,1):.1f}/64")
                cv2.putText(img, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
                ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = jpeg.tobytes()
                    self.stats = banner
                if not no_track:
                    prev = cur


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive",
                        default=str(EXAMPLE_ROOT / "depthai_models" / "descriptor64_itq_strip32_slim.tar.xz"))
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max_keypoints", type=int, default=64)
    parser.add_argument("--describe_budget", type=int, default=None,
                        help="Descriptors per frame (default: = max_keypoints). "
                        "Set below max_keypoints to show many tracked features "
                        "while describing only the oldest N each frame — the "
                        "NCE aggregates ~3000 described keypoints/s.")
    parser.add_argument("--no_tracking", action="store_true",
                        help="Detection-only descriptors: no track IDs "
                        "(motion estimator off); match lines use best-Hamming "
                        "descriptor matching between consecutive frames.")
    parser.add_argument("--describe_gate", action="store_true",
                        help="Describe only new/changed/stale keypoints "
                        "(host-side fingerprint cache); codes come back "
                        "parallel to keypoints with cached values for "
                        "unchanged ones. Static scenes run full fps with "
                        "~zero strip traffic.")
    parser.add_argument("--change_thresh", type=float, default=6.0,
                        help="Gate fingerprint change threshold (0-255 scale).")
    parser.add_argument("--refresh_every", type=int, default=30,
                        help="Gate staleness sweep period (frames).")
    parser.add_argument("--match_thresh", type=int, default=12)
    parser.add_argument("--tracker_decimate", type=int, default=2)
    parser.add_argument("--extraction_only", action="store_true",
                        help="Harris detection only: no tracking, no descriptors. "
                        "--max_keypoints sets the detection target (try 1000).")
    parser.add_argument("--fps", type=int, default=60,
                        help="extraction-only sensor rate (60: ~380 features max, "
                        "30: ~700; measured ceilings)")
    args = parser.parse_args()

    service = CaptureService(args)

    # SIGTERM/SIGINT: stop the pipeline thread and close the device cleanly.
    # An orphaned XLink session (bare kill) leaves the device mid-transfer,
    # and the next open is what hangs the firmware (watchdog reset 9001).
    def _shutdown(signum, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    service.start()

    # If the pipeline thread dies on a dead device, the HTTP server would
    # otherwise serve the last frame forever. Exit instead so an external
    # supervisor (systemd, shell loop) can restart us — process restart is
    # the only recovery from a firmware crash (USB session must drop).
    def _watchdog():
        service._thread.join()
        if service.stats.startswith("DEVICE LOST"):
            print("Pipeline died (device lost); exiting for external restart.",
                  flush=True)
            os._exit(1)

    threading.Thread(target=_watchdog, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(HTML)))
                self.end_headers()
                self.wfile.write(HTML)
                return
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg = service.latest_jpeg()
                        if jpeg is None:
                            break
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpeg)).encode()
                                         + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):  # quiet
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving on http://0.0.0.0:{args.port}/ (open / in a browser)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()


if __name__ == "__main__":
    main()
