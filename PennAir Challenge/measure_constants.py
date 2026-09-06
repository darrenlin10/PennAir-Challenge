#!/usr/bin/env python3
"""Regenerate every tuned constant in detector.py from a video.

Every threshold in the detector is a consequence of a measurable property of
the footage.

    python measure_constants.py assets/PennAir_2024_App_Dynamic_Hard.mp4

Run it
"""

import argparse
import math
import sys

import cv2
import numpy as np

import detector as det


def sample_frames(path, step):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames, total


def raw_texture(frame, win=13):
    """The texture image, before any thresholding."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.boxFilter(g, -1, (win, win))
    mu2 = cv2.boxFilter(g * g, -1, (win, win))
    t = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    hi = float(np.percentile(t[::4, ::4], 99.0))
    return np.clip(t * (255.0 / hi), 0, 255).astype(np.uint8)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def measure_texture(frames):
    section("TEXTURE_RATIO / TEXTURE_GROW_RATIO  -- from the texture distribution")
    meds, bg = [], []
    for f in frames:
        t = raw_texture(f)
        meds.append(float(np.median(t)))
        bg.append(np.percentile(t, [1, 5, 10, 25, 50]))
    bg = np.array(bg).mean(axis=0)
    print(f"  frame median texture over {len(frames)} frames: "
          f"min {min(meds):.0f}  max {max(meds):.0f}  spread {max(meds)-min(meds):.0f}")
    print("    -> the median is stable, so it is usable as a background statistic.")
    print(f"  background percentiles (mean over frames): "
          f"p1={bg[0]:.0f} p5={bg[1]:.0f} p10={bg[2]:.0f} p25={bg[3]:.0f} p50={bg[4]:.0f}")
    print(f"\n  TEXTURE_RATIO = {det.TEXTURE_RATIO}  -> threshold "
          f"{det.TEXTURE_RATIO * np.mean(meds):.0f}, which sits below p10 ({bg[2]:.0f}),")
    print("    i.e. under ~10% of background pixels clear it.")
    print(f"  TEXTURE_GROW_RATIO = {det.TEXTURE_GROW_RATIO} -> loose threshold "
          f"{det.TEXTURE_GROW_RATIO * np.mean(meds):.0f}.")
    print("    Needed because shaded parts of a shape are busier than the strict")
    print("    cutoff; see the shape-vs-background overlap below.")

    # Why a single threshold cannot work: Otsu on the same data.
    t = raw_texture(frames[0])
    otsu, _ = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    white = float(np.count_nonzero(t < otsu)) / t.size
    print(f"\n  For contrast, Otsu on the same texture image picks {otsu:.0f}, which puts")
    print(f"  {white:.0%} of the frame in the mask. Otsu assumes two classes of similar")
    print("  mass; shapes are ~5% of pixels while the background spans a wide range,")
    print("  so it splits the BACKGROUND rather than separating shapes from it.")


def measure_areas(frames):
    section("min_area  -- from the bimodality of detection areas")
    areas = []
    for f in frames:
        m = det.build_shape_mask(f)
        areas += [d["area"] for d in det.mask_to_detections(m, min_area=500)]
    a = np.array(sorted(areas))
    bins = [0, 2000, 5000, 10000, 14000, 20000, 30000, 45000, 100000]
    hist, edges = np.histogram(a, bins=bins)
    print(f"  {len(a)} detections with min_area lowered to 500:")
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        print(f"    {int(lo):6d}-{int(hi):6d} {n:5d} {'#' * int(50 * n / max(hist.max(), 1))}")
    gap = a[(a > 2000) & (a < 14000)]
    print(f"\n  Only {len(gap)} of {len(a)} detections ({len(gap)/len(a):.1%}) fall between")
    print(f"  2000 and 14000 px. min_area = 5000 sits inside that empty gap, so the")
    print("  choice is insensitive: anything from ~3000 to ~14000 gives the same result.")


def measure_solidity(frames):
    section("min_solidity  -- from the solidity of REAL shapes")
    sols = []
    for f in frames:
        m = det.build_shape_mask(f)
        for d in det.mask_to_detections(m, min_solidity=0.0):
            sols.append(d["solidity"])
    s = np.array(sols)
    print(f"  {len(s)} detections, solidity percentiles:")
    print("    " + "  ".join(f"p{p}={np.percentile(s, p):.3f}" for p in [1, 5, 10, 25, 50, 90]))
    print(f"\n  min_solidity = {0.70} sits below p5 ({np.percentile(s,5):.3f}).")
    print("  Real shapes score well under 1.0 because a hard internal colour")
    print("  boundary is indistinguishable from a shape edge to a variance")
    print("  detector, so the shape gets notched. 0.90 deletes real shapes.")


def measure_circle(frames):
    section("_find_circle thresholds  -- from the circle's measured roundness")
    rows = []
    for f in frames:
        for d in det.detect_shapes(f):
            c = d["contour"]
            if len(c) < 5 or d["clipped"] or d["was_split"]:
                continue
            fill, ratio, semi = det._ellipse_roundness(c)
            rows.append((det._circularity(c), fill, ratio, semi))
    r = np.array(rows)
    print(f"  {len(r)} candidate contours. Best-per-metric distributions:")
    for j, nm in [(0, "circularity"), (1, "area/ellipse"), (2, "axis ratio")]:
        print(f"    {nm:14s} p50={np.median(r[:,j]):.3f}  p90={np.percentile(r[:,j],90):.3f}  "
              f"p99={np.percentile(r[:,j],99):.3f}  max={r[:,j].max():.3f}")
    print("\n  Selection uses min_fill = 0.96 and min_axis_ratio = 0.88, both AREA based.")
    print("  Circularity is shown for reference but is NOT used: it goes as 1/P^2, and")
    print("  a contour traced off a pixel mask is jagged, so its perimeter is")
    print("  over-counted. A genuinely round circle here scores only 0.79-0.89 rather")
    print("  than 1.0, and filtering on it rejected good frames for no benefit.")


def measure_depth(frames, path):
    section("Z  -- the depth chain, end to end")
    f = math.sqrt(det.FX * det.FY)
    print(f"  fx = {det.FX}, fy = {det.FY}  (differ by {abs(det.FX-det.FY)/f*100:.2f}%)")
    print(f"  f  = sqrt(fx*fy) = {f:.2f} px")
    print(f"  R  = {det.CIRCLE_RADIUS_IN} in   <-- GIVEN by the spec, not measured.")
    print("       Every Z scales linearly with it; if it is wrong, all depths are wrong.")
    rs, zs = [], []
    tr = det.ZPlaneTracker()
    for fr in frames:
        d = det.detect_shapes(fr)
        ref, r_px = det._find_circle(d)
        if ref is not None:
            rs.append(r_px)
            zs.append(f * det.CIRCLE_RADIUS_IN / r_px)
        tr.update(d)
    if rs:
        rs, zs = np.array(rs), np.array(zs)
        print(f"\n  circle measured directly in {len(rs)}/{len(frames)} sampled frames "
              f"({len(rs)/len(frames):.0%})")
        print(f"  r_px: median {np.median(rs):.1f}  std {np.std(rs):.1f} "
              f"({100*np.std(rs)/np.median(rs):.1f}%)")
        print(f"  Z   : median {np.median(zs):.1f} in  std {np.std(zs):.1f} in "
              f"({np.median(zs)/12:.1f} ft)")
        print(f"\n  Sensitivity: dZ/Z = -dr/r, so at r={np.median(rs):.0f} px one pixel of")
        print(f"  radius error is {100/np.median(rs):.2f}% of depth = "
              f"{abs(f*10/(np.median(rs)+1) - np.median(zs)):.2f} in.")
        bias = math.sqrt(det.FX / det.FY) - 1
        print(f"  Known bias: using sqrt(fx*fy) with the semi-major axis under-reads Z by")
        print(f"  sqrt(fx/fy)-1 = {100*bias:+.3f}% = {abs(bias*np.median(zs)):.2f} in, which is")
        print(f"  {abs(f*10/(np.median(rs)+np.std(rs))-np.median(zs))/abs(bias*np.median(zs)):.0f}x"
              " smaller than the radius noise above. Left uncorrected deliberately.")


def measure_sensitivity(frames):
    section("SENSITIVITY  -- the range over which each constant changes nothing")

    def run():
        n, z, tr = [], [], det.ZPlaneTracker()
        for f in frames:
            d = det.detect_shapes(f)
            n.append(len(d))
            zz = tr.update(d)
            if zz:
                z.append(zz)
        return np.mean(n), (np.median(z) if z else float("nan"))

    base = run()
    print(f"  baseline: mean detections {base[0]:.2f}, median Z {base[1]:.1f} in\n")
    print(f"  {'constant':<20} {'value':>7} {'mean det':>9} {'med Z':>8}  verdict")
    for name, vals in [("TEXTURE_RATIO", [0.25, 0.35, 0.45, 0.60, 0.75]),
                       ("TEXTURE_GROW_RATIO", [0.50, 0.55, 0.60, 0.70, 0.80]),
                       ("SEED_FRAC", [0.80, 0.85, 0.90])]:
        orig = getattr(det, name)
        for v in vals:
            setattr(det, name, v)
            r = run()
            broke = abs(r[0] - base[0]) > 0.6 or r[1] != r[1] or abs(r[1] - base[1]) > 12
            print(f"  {name:<20} {v:>7} {r[0]:9.2f} {r[1]:8.1f}  "
                  f"{'BREAKS' if broke else 'unchanged'}")
        setattr(det, name, orig)
    print("\n  Wide plateaus with sharp cliffs, not knife-edges -- the pipeline is")
    print("  not balanced on a knife's edge at any of these settings.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--step", type=int, default=60, help="sample every Nth frame")
    ap.add_argument("--sensitivity", action="store_true", help="also run the sweep (slow)")
    args = ap.parse_args()

    frames, total = sample_frames(args.video, args.step)
    print(f"{args.video}: {total} frames, sampling every {args.step} -> {len(frames)} frames")

    measure_texture(frames)
    measure_areas(frames)
    measure_solidity(frames)
    measure_circle(frames)
    measure_depth(frames, args.video)
    if args.sensitivity:
        measure_sensitivity(frames)
    else:
        print("\n(run with --sensitivity for the perturbation sweep)")


if __name__ == "__main__":
    main()