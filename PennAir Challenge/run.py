"""
Harness for the PennAiR software challenge.

This file is plumbing only: it loads input, feeds it to your detector one
frame at a time, draws the results, and writes output. You should not need
to change much in here. The thinking happens in detector.py.

Usage:
    python run.py --input path/to/image.png  --output out/static.png
    python run.py --input path/to/video.mp4  --output out/dynamic.mp4
    python run.py --input path/to/video.mp4  --output out/dynamic.mp4 --show

Press q to quit early when using --show.
"""

import argparse
import os
import time

import cv2
import numpy as np

from detector import detect_shapes, estimate_xyz, ZPlaneTracker


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def draw_detections(frame, detections):
    """Draw contours, centers, and labels onto a copy of the frame.

    Args:
        frame: BGR image (H, W, 3) uint8.
        detections: list of dicts as returned by detect_shapes().

    Returns:
        A new BGR image with overlays drawn.
    """
    out = frame.copy()

    for i, det in enumerate(detections):
        contour = det["contour"]
        cx, cy = det["center"]

        cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)

        label = f"({int(round(cx))}, {int(round(cy))})"
        if det.get("xyz") is not None:
            X, Y, Z = det["xyz"]
            label = f"X{X:.1f} Y{Y:.1f} Z{Z:.1f}"

        cv2.putText(
            out, label,
            (int(round(cx)) + 10, int(round(cy)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return out


def draw_hud(frame, fps, n_shapes, z=None, z_direct=False):
    """Overlay an FPS / shape-count readout in the top-left corner."""
    text = f"{fps:5.1f} FPS | {n_shapes} shapes"
    if z is not None:
        text += f" | Z {z:.0f} in" + ("" if z_direct else " (held)")
    cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def process_image(in_path, out_path, show=False):
    frame = cv2.imread(in_path)
    if frame is None:
        raise SystemExit(f"Could not read image: {in_path}")

    t0 = time.perf_counter()
    # CHANGED: with_depth=True. A single image has no temporal context, so the
    # circle either resolves in this frame or there is no depth to report.
    detections = detect_shapes(frame, with_depth=True)
    elapsed = time.perf_counter() - t0

    out = draw_detections(frame, detections)
    print(f"Found {len(detections)} shapes in {elapsed * 1000:.1f} ms")
    if detections and all(d.get("xyz") is None for d in detections):
        print("  (no depth: no circle met the scale-reference test in this frame)")
    for i, det in enumerate(detections):
        print(f"  shape {i}: center={det['center']}, area={cv2.contourArea(det['contour']):.0f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, out)
    print(f"Wrote {out_path}")

    if show:
        cv2.imshow("result", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(in_path, out_path, show=False):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {in_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{width}x{height} @ {fps_in:.1f} fps, {total} frames")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    frame_idx = 0
    proc_times = []
    n_direct = 0
    n_with_z = 0

    # CHANGED: one tracker per video. detect_shapes() is a pure function of a
    # single frame, so anything that has to persist between frames lives here,
    # in the loop that owns the notion of a previous frame. The circle is only
    # cleanly measurable in about a third of frames on the dynamic clips --
    # it spends long stretches overlapping another shape -- and the tracker
    # carries the scene depth across those gaps using the flat-surface
    # assumption. Without it, most frames report no depth at all.
    tracker = ZPlaneTracker()

    # NOTE: this loop reads and processes ONE frame at a time, which is what
    # the challenge asks for -- the aircraft does not get the whole video up
    # front. Do not batch, do not look ahead.
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        detections = detect_shapes(frame)
        z = tracker.update(detections)
        if z is not None:
            estimate_xyz(detections, frame.shape, z_plane=z)
        proc_times.append(time.perf_counter() - t0)

        # getattr, not attribute access: if detector.py is ever out of step
        # with run.py this labels every frame "held" instead of crashing
        # mid-render after minutes of processing.
        z_direct = getattr(tracker, "last_measured_directly", False)
        n_direct += int(z_direct)
        n_with_z += int(z is not None)

        out = draw_detections(frame, detections)
        inst_fps = 1.0 / max(proc_times[-1], 1e-6)
        out = draw_hud(out, inst_fps, len(detections), z, z_direct)

        writer.write(out)
        frame_idx += 1

        if frame_idx % 60 == 0:
            mean_fps = 1.0 / (sum(proc_times) / len(proc_times))
            print(f"  frame {frame_idx}/{total}  mean {mean_fps:.1f} FPS")

        if show:
            preview = cv2.resize(out, (width // 2, height // 2))
            cv2.imshow("result", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    if proc_times:
        mean = sum(proc_times) / len(proc_times)
        worst = max(proc_times)
        print(f"\nProcessed {frame_idx} frames")
        print(f"  mean   {1.0 / mean:6.1f} FPS  ({mean * 1000:.1f} ms/frame)")
        print(f"  worst  {1.0 / worst:6.1f} FPS  ({worst * 1000:.1f} ms/frame)")
        print(f"  depth  {n_with_z / frame_idx:5.0%} of frames "
              f"({n_direct / frame_idx:.0%} measured directly from the circle, "
              f"the rest carried by ZPlaneTracker)")
        print(f"  --> quote the mean in your README; efficiency is graded")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="PennAiR challenge harness")
    ap.add_argument("--input", required=True, help="image or video path")
    ap.add_argument("--output", required=True, help="where to write the result")
    ap.add_argument("--show", action="store_true", help="live preview window")
    args = ap.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    if ext in IMAGE_EXTS:
        process_image(args.input, args.output, args.show)
    else:
        process_video(args.input, args.output, args.show)


if __name__ == "__main__":
    main()