"""
Shape detection and monocular depth estimation for the PennAiR 2026 challenge.

PIPELINE
--------
    build_shape_mask()    frame  -> black/white mask, white where the shapes are
    mask_to_detections()  mask   -> list of shapes with outlines and centres
    estimate_xyz()        shapes -> 3D position of each shape, in inches
    ZPlaneTracker         keeps the depth estimate alive between video frames

`measure_constants.py` recomputes every tuned constant from a video.
"""

import math

import cv2

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

K = np.array([
    [2564.3186869, 0.0,           0.0],
    [0.0,          2569.70273111, 0.0],
    [0.0,          0.0,           1.0],
])

# circle radius is 10 inches
CIRCLE_RADIUS_IN = 10.0

# Pull the two focal lengths out of K.
FX = float(K[0, 0])
FY = float(K[1, 1])

# K gives (0, 0), which is not credible. None means use the image centre.
PRINCIPAL_POINT = None

# Offset between the mask boundary and the true shape edge. Uncalibrated.
RADIUS_BIAS_PX = 0.0


# =============================================================================
# CONSTANTS MEASURED
# =============================================================================

# Mask cutoff, as a fraction of the frame's median texture, so it follows the
# footage instead of being fixed.
TEXTURE_RATIO = 0.45

# Looser cutoff for hysteresis: only kept where it touches a strict-pass pixel.
TEXTURE_GROW_RATIO = 0.60

# Distance-transform slice level when splitting touching shapes. Well above the
# usual tutorial value of 0.5, which gives up on all but slight overlaps.
SEED_FRAC = 0.85


# =============================================================================
# BUILD THE MASK
# =============================================================================

def build_shape_mask(frame, win=13, min_area=800, debug=None):
    """Colour photo -> black-and-white mask, white = shape.

    Shapes are smooth and the background is not, so the discriminator is local
    variance over a win x win neighbourhood.
    """

    # 1. MEASURE HOW NOISY EACH NEIGHBOURHOOD IS ------------------------------

    # float32 because we are about to square these and uint8 would overflow.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # variance = mean of squares - square of mean, and boxFilter gives a local
    # mean in O(1) per pixel. So two box filters give variance everywhere.
    ksize = (win, win)
    mu = cv2.boxFilter(gray, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.boxFilter(gray * gray, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    # Clamp at zero: rounding can make variance slightly negative, and sqrt of
    # a negative is NaN.
    var = np.maximum(mu2 - mu * mu, 0.0)
    texture = np.sqrt(var)

    # 2. RESCALE THE TEXTURE IMAGE TO 0-255 -----------------------------------

    # Scale against a high percentile, not the max, so one glare spot cannot
    # crush everything else into the bottom few levels. Subsampled for speed.
    hi = float(np.percentile(texture[::4, ::4], 99.0))

    # Perfectly flat frame: nothing to find, and dividing by hi would blow up.
    if hi < 1e-3:
        return np.zeros(gray.shape, np.uint8)

    # Dark where smooth (the shapes), bright where noisy (the background).
    texture_u8 = np.clip(texture * (255.0 / hi), 0, 255).astype(np.uint8)

    # 3. DECIDE WHICH PIXELS ARE SMOOTH ENOUGH TO BE A SHAPE ------------------

    # Not Otsu: it assumes two classes of similar size, but the shapes are a
    # small minority, so it splits the background instead and floods the mask.
    # The background dominates by area, so its median describes the background.
    median_texture = float(np.median(texture_u8))

    # BINARY_INV = white where BELOW the cutoff, i.e. the smooth pixels.
    _, mask = cv2.threshold(texture_u8, TEXTURE_RATIO * median_texture, 255,
                            cv2.THRESH_BINARY_INV)

    # Hysteresis, as in Canny. A shape is busiest where its own shading is
    # steepest, so shaded corners fail the strict cutoff. Keep loose-cutoff
    # regions only where they touch a strict-cutoff seed. This also drops
    # strict-mask speckle that has no substantial core.
    if TEXTURE_GROW_RATIO > TEXTURE_RATIO:
        _, loose = cv2.threshold(texture_u8, TEXTURE_GROW_RATIO * median_texture,
                                 255, cv2.THRESH_BINARY_INV)

        # Open the seeds first, or a single stray pixel seeds a huge region.
        seeds = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        # Number each separate white island in `loose`; 0 is the background.
        _, labels = cv2.connectedComponents(loose)

        # Keep only the islands containing at least one seed.
        keep = np.unique(labels[seeds > 0])
        keep = keep[keep != 0]
        if keep.size:
            mask = np.isin(labels, keep).astype(np.uint8) * 255

    # Capture before the cleanup, which fills blobs solid and would hide
    # whether the threshold itself was clean.
    if debug is not None:
        debug["texture"] = texture_u8
        debug["raw_mask"] = mask.copy()

    # 4. CLEAN UP THE MASK ----------------------------------------------------

    # OPEN (erode then dilate) removes specks; CLOSE (dilate then erode) fills
    # holes. RECT for the close because it is separable and so much faster, and
    # closing leaves a convex blob's boundary where it was.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # 5. FILL EACH BLOB SOLID AND DROP THE TINY ONES --------------------------

    # RETR_EXTERNAL = outer outlines only, ignoring holes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Repainting each outline filled also removes interior holes for free.
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, thickness=cv2.FILLED)

    # 6. GIVE BACK THE BORDER THE MEASUREMENT ATE -----------------------------

    # A window on a shape's edge spans both sides and reads as noisy, so every
    # shape comes out shrunk by about win/2. ELLIPSE, not RECT: this dilate does
    # change the final outline, and a square would inflate the circle's radius.
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win))
    clean = cv2.dilate(clean, k_grow)

    if debug is not None:
        debug["filled_mask"] = clean.copy()

    return clean


# =============================================================================
# SPLITTING TWO TOUCHING SHAPES APART
# =============================================================================

def _split_blob(contour, seed_frac=SEED_FRAC, min_piece_area=500):
    """Split one blob that is really two shapes touching.

    The distance transform (distance from each white pixel to the nearest black
    one) makes one hill per shape, with a dip at the waist between them. Slice
    above that dip to get one seed per shape, then watershed grows them back.
    """

    # Work in a small cropped canvas rather than the full frame, for speed.
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    local = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(local, [contour - [x - pad, y - pad]], -1, 255, cv2.FILLED)

    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [contour]

    _, seeds = cv2.threshold(dist, seed_frac * dist.max(), 255, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)

    # One island means one genuine shape; leave it alone.
    n_seeds, labels = cv2.connectedComponents(seeds)
    if n_seeds <= 2:
        return [contour]

    # Watershed markers: 0 = decide this, 1 = background, 2+ = known regions.
    markers = labels.astype(np.int32) + 1
    markers[local == 0] = 1
    markers[(local > 0) & (seeds == 0)] = 0
    cv2.watershed(cv2.cvtColor(local, cv2.COLOR_GRAY2BGR), markers)

    # Back to outlines in full-frame coordinates.
    pieces = []
    for k in range(2, n_seeds + 1):
        piece = np.uint8(markers == k) * 255
        found, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for p in found:
            if cv2.contourArea(p) >= min_piece_area:
                pieces.append(p + [x - pad, y - pad])

    return pieces or [contour]


# =============================================================================
# STEP 2 OF THE PIPELINE: MASK -> LIST OF SHAPES
# =============================================================================

def mask_to_detections(mask, min_area=5000, min_solidity=0.70, border_margin=3):
    """Mask -> list of dicts with contour, center, xyz, area, solidity, clipped,
    was_split. xyz stays None until estimate_xyz() fills it in."""

    H, W = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:

        # min_area also rejects background: detection areas are bimodal, with
        # the shapes an order of magnitude larger than any leftover artifact.
        if len(c) < 3 or cv2.contourArea(c) < min_area:
            continue

        # One blob may be two shapes touching, so try to separate it first.
        parts = _split_blob(c)
        was_split = len(parts) > 1

        for p in parts:
            area = cv2.contourArea(p)
            if area < min_area:
                continue

            # Solidity = area / convex hull area. Near 1.0 for a solid convex
            # blob, low for a stringy or dumbbell-shaped one. Set loose, because
            # a hard internal colour boundary notches real shapes here. Checked
            # AFTER the split: a merged pair scores low and would be deleted
            # before it could be separated.
            hull_area = cv2.contourArea(cv2.convexHull(p))
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            # Centroid from image moments: m10/m00 is the mean x, m01/m00 the
            # mean y. Beats the bounding box centre on triangles and the like.
            M = cv2.moments(p)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Running off the frame means the centroid belongs to the visible
            # part only. Flagged rather than deleted; it is still a detection.
            x, y, w, h = cv2.boundingRect(p)
            clipped = (x <= border_margin or y <= border_margin
                       or x + w >= W - border_margin or y + h >= H - border_margin)

            detections.append({
                "contour": p,
                "center": (float(cx), float(cy)),
                "xyz": None,
                "area": float(area),
                "solidity": float(solidity),
                "clipped": bool(clipped),
                "was_split": bool(was_split),
            })

    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections


# =============================================================================
# FINDING THE CIRCLE, WHICH IS OUR ONLY REAL-WORLD RULER
# =============================================================================

def _circularity(contour):
    """4*pi*Area / Perimeter^2. 1.0 for a perfect circle, less otherwise."""
    P = cv2.arcLength(contour, True)
    if P <= 0:
        return 0.0
    return 4.0 * math.pi * cv2.contourArea(contour) / (P * P)


def _ellipse_roundness(contour):
    """Returns (fill, axis_ratio, semi_major_px) from a fitted ellipse.

    Both are AREA based, unlike _circularity, whose perimeter term is inflated
    by the staircased outline of a pixel mask.
    """
    # fitEllipse returns ((cx, cy), (width, height), rotation).
    (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)

    # pi*a*b on the semi-axes, so the full widths give a /4.
    ellipse_area = math.pi * axis_a * axis_b / 4.0
    if ellipse_area <= 0:
        return 0.0, 0.0, 0.0

    return (cv2.contourArea(contour) / ellipse_area,
            min(axis_a, axis_b) / max(axis_a, axis_b),
            max(axis_a, axis_b) / 2.0)


def _find_circle(detections, min_fill=0.96, min_axis_ratio=0.88):
    """Pick the circle. Returns (detection, radius_px) or (None, None).

    Deliberately strict: this one measurement scales every distance in the
    frame, so reporting nothing beats picking the wrong object.
    """
    best, best_r, best_score = None, None, -1.0

    for d in detections:
        # A clipped outline is truncated and a split one has an artificial cut
        # across it. Neither has a trustworthy radius.
        if d.get("clipped") or d.get("was_split"):
            continue

        c = d["contour"]
        if len(c) < 5:                         # fitEllipse needs 5+ points
            continue

        fill, axis_ratio, semi_major = _ellipse_roundness(c)
        if fill < min_fill or axis_ratio < min_axis_ratio:
            continue

        if fill > best_score:
            best, best_r, best_score = d, semi_major, fill

    return best, best_r


# =============================================================================
# STEP 3 OF THE PIPELINE: PIXELS -> INCHES
# =============================================================================

def estimate_xyz(detections, image_shape, fx=FX, fy=FY, cx=None, cy=None,
                 circle_radius_in=CIRCLE_RADIUS_IN,
                 radius_bias_px=RADIUS_BIAS_PX, z_plane=None):
    """Estimate x, y, z in inches from the pinhole model.

        r = f * R / Z   (similar triangles)  ->  Z = f * R / r
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """

    H, W = image_shape[:2]

    # K's principal point is not credible, so use the image centre. This moves
    # X and Y only; Z depends on the focal length and radius alone.
    if cx is None:
        cx = PRINCIPAL_POINT[0] if PRINCIPAL_POINT else W / 2.0
    if cy is None:
        cy = PRINCIPAL_POINT[1] if PRINCIPAL_POINT else H / 2.0

    # fx and fy differ slightly (non-square pixels) and the radius is measured
    # along an arbitrary axis, so neither alone is right. Geometric mean.
    f = math.sqrt(fx * fy)

    ref = None
    if z_plane is not None:
        Z = float(z_plane)
    else:
        ref, r_px = _find_circle(detections)

        # No ruler, no distances. Say so rather than guessing.
        if ref is None:
            for d in detections:
                d["xyz"] = None
            return detections

        r_px -= radius_bias_px
        if r_px <= 0:
            for d in detections:
                d["xyz"] = None
            return detections

        # Semi-major axis, not the mean of the two. Under tilt the long axis
        # lies along the rotation axis where points stay at depth Z, so it still
        # obeys r = f*R/Z exactly; the short axis is foreshortened.
        Z = f * circle_radius_in / r_px

    for d in detections:
        # Flat surface, so every shape shares this depth.
        d["z_plane"] = float(Z)

        # Depth is still fine when clipped, but the centroid is the visible
        # fragment's, so X and Y would be wrong. None means "not trustworthy".
        if d.get("clipped"):
            d["xyz"] = None
            continue

        u, v = d["center"]
        d["xyz"] = (float((u - cx) * Z / fx), float((v - cy) * Z / fy), float(Z))

    if ref is not None:
        ref["is_scale_reference"] = True

    return detections


def detect_shapes(frame, with_depth=False):
    """Run the whole pipeline on one frame.

    Stateless: anything that must persist across frames lives in the caller's
    loop, so a still image can be tested alone and two videos cannot interfere.
    """
    mask = build_shape_mask(frame)
    detections = mask_to_detections(mask)
    if with_depth:
        detections = estimate_xyz(detections, frame.shape)
    return detections


# =============================================================================
# Check depth
# =============================================================================

class ZPlaneTracker:
    """Remembers the scene depth between frames. Create one per video.

    The circle is only cleanly measurable in a minority of frames, and the gaps
    are too long to reuse a stale depth while the camera moves. But all the
    shapes share one plane, so a single good measurement fixes every other
    shape's real size and any of them can then be the ruler:

        real_area = pixel_area * (Z / f)^2
        Z         = f * sqrt(real_area / pixel_area)

    Usage:
        tracker = ZPlaneTracker()
        for frame in frames:
            dets = detect_shapes(frame)
            z = tracker.update(dets)
            if z is not None:
                estimate_xyz(dets, frame.shape, z_plane=z)

    Limits: nearest-centre matching is wrong if two shapes cross, learned sizes
    inherit their measurement's error, and a frame with no clear shape is lost.
    """

    def __init__(self, fx=FX, fy=FY, circle_radius_in=CIRCLE_RADIUS_IN,
                 match_px=60.0, max_stale=90, max_rel_step=0.03):
        self.f = math.sqrt(fx * fy)
        self.circle_radius_in = circle_radius_in
        self.match_px = match_px          # max pixels a shape may move per frame
        self.max_stale = max_stale        # frames we will reuse an old depth for
        self.max_rel_step = max_rel_step  # max fractional depth change per frame

        self.sizes = {}                   # track id -> real area, square inches
        self.tracks = {}                  # track id -> where it was last seen
        self._next_id = 0

        self.last_z = None
        self.stale = 0

        # True when update() measured the circle for real. Callers need it:
        # estimate_xyz cannot mark a reference when simply handed a depth.
        self.last_measured_directly = False

    def _match(self, detections):
        """Match this frame's shapes to last frame's by nearest centre."""
        ids, used = [], set()
        for d in detections:
            cx, cy = d["center"]
            best, best_dist = None, self.match_px

            # Closest previous track that nothing else has claimed.
            for tid, (px, py) in self.tracks.items():
                if tid in used:
                    continue
                dist = math.hypot(cx - px, cy - py)
                if dist < best_dist:
                    best, best_dist = tid, dist

            # Nothing close enough, so this is a shape we have not seen before.
            if best is None:
                best = self._next_id
                self._next_id += 1

            used.add(best)
            self.tracks[best] = (cx, cy)
            ids.append(best)

        return ids

    def update(self, detections):
        """Return the scene depth in inches for this frame, or None."""

        ids = self._match(detections)
        z, trusted = None, False

        # Option 1, best: the circle is clearly visible.
        ref, r_px = _find_circle(detections)
        if ref is not None and r_px > 0:
            z, trusted = self.f * self.circle_radius_in / r_px, True

        # Option 2: infer it from a shape we sized on an earlier good frame.
        if z is None:
            votes = []
            for tid, d in zip(ids, detections):
                if d.get("clipped") or d.get("was_split"):
                    continue
                area = d.get("area", 0.0)
                if tid in self.sizes and area > 0:
                    cand = self.f * math.sqrt(self.sizes[tid] / area)

                    # Screen against the last depth. Nearest-centre matching
                    # does confuse shapes, and a confused identity gives a
                    # wildly wrong depth rather than a slightly wrong one.
                    if self.last_z is None or \
                            abs(cand - self.last_z) <= self.max_rel_step * self.last_z:
                        votes.append(cand)

            # Median so one bad shape cannot drag the answer around.
            if votes:
                z = float(np.median(votes))

        self.last_measured_directly = trusted

        # Option 3: nothing worked, so reuse the old value for a while.
        if z is None:
            self.stale += 1
            return self.last_z if self.stale <= self.max_stale else None

        # Learn sizes ONLY from a direct measurement. Learning from an inferred
        # depth feeds the estimate back into itself and drifts badly.
        if trusted:
            for tid, d in zip(ids, detections):
                if d.get("clipped") or d.get("was_split"):
                    continue
                area = d.get("area", 0.0)
                if area > 0:
                    prev = self.sizes.get(tid)
                    new = area * (z / self.f) ** 2
                    # Blend, so one noisy frame does not throw the size off.
                    self.sizes[tid] = new if prev is None else 0.7 * prev + 0.3 * new

        self.last_z, self.stale = z, 0
        return z