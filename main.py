"""Binary feature descriptor demo for OAK-1 (RVC2).

Tracks keypoints with the on-device FeatureTracker, describes 32x32 grayscale
patches with a custom NNArchive (512-bit raw descriptor + 64-bit compressed
code), and visualizes keypoints plus temporal code stability on the DepthAI
Visualizer.
"""

import cv2
import depthai as dai
import numpy as np
from dotenv import load_dotenv

from core.arguments import initialize_argparser
from core.packer import match_codes
from core.pipeline import FeaturePipeline

load_dotenv(override=True)

_, args = initialize_argparser()


def track_color(track_id):
    """Deterministic BGR color per track ID."""
    return (
        int((37 * track_id + 80) % 255),
        int((97 * track_id + 40) % 255),
        int((61 * track_id + 160) % 255),
    )


def main():
    visualizer = dai.RemoteConnection(httpPort=8082)

    try:
        tw, th = (int(v) for v in args.tracker_size.lower().split("x"))
        fp = FeaturePipeline(
            archive_path=args.archive,
            max_keypoints=args.max_keypoints,
            describe_budget=args.describe_budget,
            fps_limit=args.fps_limit,
            device=args.device,
            media_path=args.media_path,
            publish_full_desc=not args.no_full_desc,
            detector=args.detector,
            nn_nodes=args.nn_nodes,
            tracker_size=(tw, th),
            tracker_decimate=args.tracker_decimate,
            motion_estimator=args.motion_estimator,
            autotune=not args.no_autotune,
            color=args.color,
            tracking=not args.no_tracking,
            describe_gate=args.describe_gate,
            change_thresh=args.change_thresh,
            refresh_every=args.refresh_every,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    with fp:
        print(f"Platform: {fp.platform}")
        visualizer.registerPipeline(fp.pipeline)
        keypoints_topic = visualizer.addTopic("Keypoints", "images")

        prev_codes = None
        while fp.pipeline.isRunning():
            if visualizer.waitKey(1) == ord("q"):
                print("Got q key from the remote connection!")
                break

            try:
                result = fp.next_frame(timeout=1.0)
            except dai.MessageQueue.QueueException:
                break
            except RuntimeError as e:
                # Device died or wedged (firmware crash/hang); the message
                # explains recovery. Exit so the process fully releases USB.
                print(e)
                break
            if result is None:
                continue

            frame, keypoints, codes, _ = result
            vis = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.ndim == 2
                else frame.copy()
            )
            for kp in keypoints:
                **********(
                    vis,
                    (int(round(kp.x)), int(round(kp.y))),
                    3,
                    track_color(kp.track_id if kp.track_id >= 0 else 0),
                    -1,
                    cv2.LINE_AA,
                )

            stats = f"keypoints: {len(keypoints)}"
            if prev_codes is not None and len(codes) and len(prev_codes):
                matches = match_codes(codes, prev_codes)
                if matches:
                    mean_dist = float(np.mean([d for _, _, d in matches]))
                    stats += (
                        f" | matched: {len(matches)}"
                        f" | mean hamming: {mean_dist:.1f}"
                    )
            print(stats)
            cv2.putText(
                vis, stats, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )

            img = dai.ImgFrame()
            img.setCvFrame(vis, dai.ImgFrame.Type.BGR888i)
            keypoints_topic.send(img)
            prev_codes = codes


if __name__ == "__main__":
    main()
