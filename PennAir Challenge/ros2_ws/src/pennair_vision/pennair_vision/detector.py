"""
Shape detection and monocular depth estimation for the PennAiR 2024 challenge.

THE CORE IDEA
-------------
We must find coloured shapes lying on a background, and it has to keep working
when the background changes colour. So we cannot key on colour at all.

What separates the shapes from the background is TEXTURE, not colour. The grass
or asphalt is noisy at the pixel level: neighbouring pixels differ a lot. The
shapes are smooth: even the ones with a colour gradient across them change only
gradually from one pixel to the next. So for every pixel we ask "how much do the
pixels around me disagree with each other?", and the answer is small inside a
shape and large on the background. That measure is the local standard deviation.

PIPELINE
--------
    build_shape_mask()    frame  -> black/white mask, white where the shapes are
    mask_to_detections()  mask   -> list of shapes with outlines and centres
    estimate_xyz()        shapes -> 3D position of each shape, in inches
    ZPlaneTracker         keeps the depth estimate alive between video frames

Every threshold below was measured, not guessed. `measure_constants.py`
recomputes them all from a video and prints the reasoning.
"""

# math: only for sqrt, pi and hypot.
import math

# cv2 is OpenCV, the computer-vision library that does the image processing.
import cv2
# numpy handles images as big arrays of numbers, which is what an image is:
# a grid of height x width x 3 (blue, green, red) values from 0 to 255.
import numpy as np


# =============================================================================
# CONSTANTS GIVEN BY THE CHALLENGE
# =============================================================================

# The camera intrinsic matrix. It describes how the camera turns a point in the
# real world into a pixel on the image. The two numbers on the diagonal are the
# focal lengths in PIXELS: roughly "how many pixels wide does a one-unit object
# at one-unit distance appear". We need them to convert pixels into inches.
K = np.array([
    [2564.3186869, 0.0,           0.0],
    [0.0,          2569.70273111, 0.0],
    [0.0,          0.0,           1.0],
])

# The real radius of the circle in the scene, in inches. This is the ONLY real
# world measurement we are given, so the circle is what sets the scale for
# everything else. If this number is wrong, every distance we output is wrong
# by the same factor.
CIRCLE_RADIUS_IN = 10.0

# Pull the two focal lengths out of K. K[0,0] is the horizontal one (fx) and
# K[1,1] the vertical one (fy). They differ slightly, which just means the
# camera's pixels are not perfectly square.
FX = float(K[0, 0])
FY = float(K[1, 1])

# The "principal point" is where the camera's optical axis hits the image; on a
# real camera it is near the middle of the picture. The K we were given says
# (0, 0), which would mean the lens points at the top-left CORNER of the sensor.
# That is not physically sensible, so we treat it as missing and substitute the
# image centre when we need it. None here means "work it out from the image
# size"; set it to an (x, y) tuple to override.
PRINCIPAL_POINT = None

# The mask we build is slightly larger or smaller than the true shape because of
# the morphology in build_shape_mask(). If you calibrate that offset, put it
# here and it is subtracted from the measured circle radius. Left at zero
# because it has not been calibrated against real footage.
RADIUS_BIAS_PX = 0.0


# =============================================================================
# CONSTANTS MEASURED FROM THE FOOTAGE  (see measure_constants.py)
# =============================================================================

# A pixel joins the mask if its texture is below TEXTURE_RATIO x (the median
# texture of the whole frame). Using a FRACTION OF THE MEDIAN rather than a
# fixed number means the threshold follows the footage: rough asphalt and
# smooth grass both work without retuning.
TEXTURE_RATIO = 0.45

# Second, looser threshold used for "hysteresis" (explained at the point of
# use). Pixels below this only count if they are touching a pixel that already
# passed the strict threshold above.
TEXTURE_GROW_RATIO = 0.60

# Used when splitting two touching shapes apart. Explained in _split_blob().
SEED_FRAC = 0.85


# =============================================================================
# STEP 1 OF THE PIPELINE: BUILD THE MASK
# =============================================================================

def build_shape_mask(frame, win=13, min_area=800, debug=None):
    """Turn a colour photo into a black-and-white mask: white = shape.

    Args:
        frame:    the colour image, a numpy array of shape (height, width, 3).
        win:      the size of the neighbourhood, in pixels, over which we
                  measure "how noisy is it here". 13 means a 13x13 square.
        min_area: blobs smaller than this many pixels are thrown away as noise.
        debug:    pass an empty dict {} to have the intermediate images stored
                  in it, so you can save them and look at what went wrong.

    Returns:
        A single-channel image the same width and height as the input, where
        every pixel is either 0 (background) or 255 (shape).
    """

    # ------------------------------------------------------------------
    # 1. MEASURE HOW NOISY EACH NEIGHBOURHOOD IS
    # ------------------------------------------------------------------

    # Colour is irrelevant to us, so collapse the 3 colour channels down to one
    # brightness channel. float32 because we are about to square these numbers
    # and squaring a uint8 (max 255) would overflow and wrap around.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # We want the standard deviation of the pixels inside a win x win box
    # centred on every pixel. Doing that literally would be very slow. Instead
    # use the identity:
    #
    #     variance = (average of the squares) - (square of the average)
    #
    # cv2.boxFilter computes a local AVERAGE over a box, and it does so in
    # constant time per pixel no matter how big the box is. So two box filters
    # give us the variance everywhere at once.
    ksize = (win, win)

    # mu = the local average brightness.
    mu = cv2.boxFilter(gray, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    # mu2 = the local average of the SQUARED brightness.
    mu2 = cv2.boxFilter(gray * gray, -1, ksize, normalize=True,
                        borderType=cv2.BORDER_REFLECT)

    # Now apply the identity. np.maximum(..., 0.0) clamps at zero: the maths
    # says variance can never be negative, but floating point rounding can push
    # it a hair below, and the sqrt of a negative number is NaN.
    var = np.maximum(mu2 - mu * mu, 0.0)

    # Standard deviation is the square root of variance. We use it instead of
    # variance because it is in the same units as brightness, which makes the
    # numbers easier to reason about.
    texture = np.sqrt(var)

    # ------------------------------------------------------------------
    # 2. RESCALE THE TEXTURE IMAGE TO 0-255
    # ------------------------------------------------------------------

    # cv2.threshold wants 8-bit input, so squash our float values into 0-255.
    # We scale against the 99th percentile rather than the maximum: if a single
    # bright glare spot has a huge texture value, scaling by the maximum would
    # crush every other pixel down into the bottom few levels and destroy all
    # the detail we care about.
    #
    # texture[::4, ::4] takes every 4th row and column. Computing a percentile
    # requires sorting, which is slow on 2 million pixels; sampling one
    # sixteenth of them gives the same answer to well within what we need.
    hi = float(np.percentile(texture[::4, ::4], 99.0))

    # If hi is essentially zero the whole frame is perfectly flat, which means
    # there is nothing to find. Bail out rather than divide by zero.
    if hi < 1e-3:
        return np.zeros(gray.shape, np.uint8)

    # Scale so the 99th percentile lands at 255, clip anything above, convert
    # to 8-bit. texture_u8 is now a greyscale picture of "business": dark where
    # the image is smooth (the shapes), bright where it is noisy (background).
    texture_u8 = np.clip(texture * (255.0 / hi), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 3. DECIDE WHICH PIXELS ARE SMOOTH ENOUGH TO BE A SHAPE
    # ------------------------------------------------------------------

    # We need a cutoff. The obvious choice is Otsu's method, which picks a
    # cutoff automatically, and it FAILS BADLY here, so it is worth saying why.
    #
    # Otsu assumes the image contains two groups of roughly equal size and finds
    # the split between them. In our frames the shapes are only about 5% of the
    # pixels, while the background's texture is spread over a wide range. Otsu
    # therefore finds it "cheaper" to cut the BACKGROUND in half than to
    # separate the shapes from it. On the challenge footage it picks 112, which
    # puts 59% of the whole frame into the mask, and everything downstream then
    # collapses.
    #
    # Instead: because the background covers most of the frame, the MEDIAN
    # texture of the frame IS a measurement of typical background texture. The
    # shapes sit far below it. So use a fixed fraction of the median.
    #
    # The assumption to remember: this needs the background to dominate. If
    # shapes ever covered more than roughly a third of the image, the median
    # would stop describing the background and this would break.
    median_texture = float(np.median(texture_u8))

    # THRESH_BINARY_INV means "white where the value is BELOW the cutoff".
    # We want the low-texture (smooth) pixels, hence the inverted version.
    _, mask = cv2.threshold(texture_u8, TEXTURE_RATIO * median_texture, 255,
                            cv2.THRESH_BINARY_INV)

    # --- hysteresis: rescue the shaded parts of a shape ---
    #
    # Problem: a shape with strong shading is at its "busiest" exactly where the
    # shading changes fastest, e.g. the vignetted corners of the white
    # trapezoid. Those corners fail the strict cutoff and get chopped off.
    #
    # We cannot simply raise the cutoff, because the two populations overlap:
    # the shaded shape pixels and the smoothest background pixels sit at the
    # same texture values. Any cutoff high enough to keep the corners also lets
    # in a big chunk of background.
    #
    # The way out is hysteresis, the same trick the Canny edge detector uses.
    # Take a second, more generous cutoff. Keep a generous-cutoff region ONLY if
    # it touches a region that already passed the strict cutoff. A shaded corner
    # is attached to the solidly-detected middle of its shape, so it survives.
    # An isolated smooth speck of asphalt touches no shape, so it does not.
    if TEXTURE_GROW_RATIO > TEXTURE_RATIO:

        # The generous mask: everything that might be part of a shape.
        _, loose = cv2.threshold(texture_u8, TEXTURE_GROW_RATIO * median_texture,
                                 255, cv2.THRESH_BINARY_INV)

        # The confident mask, cleaned up. MORPH_OPEN erodes then dilates, which
        # deletes anything thinner than the 9x9 brush while leaving big regions
        # alone. Without this, a single stray background pixel would count as a
        # "seed" and could drag in a huge connected region of loose pixels.
        seeds = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        # connectedComponents numbers each separate white island in `loose`:
        # every pixel of island 1 gets the value 1, island 2 gets 2, and so on,
        # with 0 meaning black background.
        _, labels = cv2.connectedComponents(loose)

        # Which island numbers have at least one confident seed inside them?
        keep = np.unique(labels[seeds > 0])
        keep = keep[keep != 0]                 # 0 is the black background

        # Rebuild the mask from only those islands.
        if keep.size:
            mask = np.isin(labels, keep).astype(np.uint8) * 255

    # Stash the intermediates if the caller asked for them. Do this BEFORE the
    # cleanup below, because the cleanup fills every blob in solid and would
    # hide whether the threshold itself was clean or patchy.
    if debug is not None:
        debug["texture"] = texture_u8
        debug["raw_mask"] = mask.copy()

    # ------------------------------------------------------------------
    # 4. CLEAN UP THE MASK
    # ------------------------------------------------------------------

    # Morphology works by sliding a small shape (a "structuring element", think
    # of it as a brush) over the image.
    #   OPEN  = erode then dilate -> removes small white specks.
    #   CLOSE = dilate then erode -> fills small black holes.

    # Round brush for the open, so we do not carve corners into the specks.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Square brush for the close. A square is separable, which makes OpenCV's
    # implementation far faster (about 1.6 ms versus 9 ms at 1080p), and closing
    # a convex blob puts its outer boundary back exactly where it was, so the
    # brush's shape does not leak into the result.
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)     # kill speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)  # fill holes

    # ------------------------------------------------------------------
    # 5. FILL EACH BLOB SOLID AND DROP THE TINY ONES
    # ------------------------------------------------------------------

    # findContours traces the outline of every white island.
    #   RETR_EXTERNAL      = only the outer outlines, ignore holes inside them.
    #   CHAIN_APPROX_SIMPLE = store corner points rather than every pixel.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Start from a blank image and paint each surviving outline back in, filled.
    # This removes any interior holes for free.
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, thickness=cv2.FILLED)

    # ------------------------------------------------------------------
    # 6. GIVE BACK THE BORDER THE MEASUREMENT ATE
    # ------------------------------------------------------------------

    # A win x win window sitting on a shape's edge covers both the shape and the
    # background, so it sees a big brightness jump and reports high texture.
    # Every shape therefore comes out shrunk by roughly half a window. Dilating
    # by the same amount puts the boundary back.
    #
    # ELLIPSE here, not RECT: unlike the close above, this dilate DOES change
    # the final outline. A square brush would push the circle's outline outward
    # at the diagonals and inflate the radius, which would corrupt the depth
    # calculation later.
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win))
    clean = cv2.dilate(clean, k_grow)

    if debug is not None:
        debug["filled_mask"] = clean.copy()

    return clean


# =============================================================================
# SPLITTING TWO TOUCHING SHAPES APART
# =============================================================================

def _split_blob(contour, seed_frac=SEED_FRAC, min_piece_area=500):
    """Split one blob that is really two shapes touching each other.

    When two shapes overlap they become one white island, and RETR_EXTERNAL
    returns a single outline wrapped around both. Depending on how much they
    overlap you then get one of two wrong answers: barely-touching shapes get
    deleted by the solidity filter, and heavily-overlapping ones come back as a
    single detection whose centre sits in the gap between them.

    THE FIX: the "distance transform". For every white pixel it computes how far
    that pixel is from the nearest black pixel. Inside a single round shape this
    forms one hill peaking at the centre. Inside a figure-of-eight it forms TWO
    hills with a dip at the waist. Slice near the top of the hills and you get
    one island per shape, then watershed grows them back out.
    """

    # Work in a small cropped canvas rather than the full frame, for speed.
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    local = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(local, [contour - [x - pad, y - pad]], -1, 255, cv2.FILLED)

    # Build the hills.
    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [contour]

    # Slice the hills at 85% of the tallest peak.
    #
    # Most tutorials use 0.5 and that does NOT work here: at 50% the waist
    # between two overlapping circles is still above the water line, so the two
    # hills stay joined and you get a single island. 0.85 is where they finally
    # separate.
    _, seeds = cv2.threshold(dist, seed_frac * dist.max(), 255, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)

    # One island means one genuine shape; leave it alone.
    n_seeds, labels = cv2.connectedComponents(seeds)
    if n_seeds <= 2:
        return [contour]

    # Same watershed marker convention as above.
    markers = labels.astype(np.int32) + 1
    markers[local == 0] = 1
    markers[(local > 0) & (seeds == 0)] = 0
    cv2.watershed(cv2.cvtColor(local, cv2.COLOR_GRAY2BGR), markers)

    # Convert each region back to an outline in full-frame coordinates.
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
    """Turn the black-and-white mask into a list of described shapes.

    Args:
        mask:         output of build_shape_mask().
        min_area:     ignore blobs smaller than this many pixels.
        min_solidity: ignore blobs less "solid" than this (see below).

    Returns a list of dicts, one per shape, each with:
        "contour"  the outline, as an array of points
        "center"   (x, y) centre in pixels
        "xyz"      None for now; estimate_xyz() fills this in
        plus "area", "solidity", "clipped" and "was_split".
    """

    H, W = mask.shape[:2]

    # Trace the outline of every white island in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:

        # A polygon needs at least 3 points, and anything tiny is not a shape.
        # min_area also does the job of rejecting background: detection areas
        # are sharply bimodal, with background artifacts under 2000 pixels and
        # real shapes above 20000.
        if len(c) < 3 or cv2.contourArea(c) < min_area:
            continue

        # One blob may be two shapes touching, so try to separate it first.
        parts = _split_blob(c)
        was_split = len(parts) > 1

        for p in parts:
            area = cv2.contourArea(p)
            if area < min_area:
                continue

            # "Solidity" = the blob's area divided by the area of its convex
            # hull. The convex hull is the shape you get by stretching a rubber
            # band around the outline. A solid convex blob scores near 1.0; a
            # stringy or dumbbell-shaped one scores much lower.
            #
            # The threshold is 0.70, not the 0.90 you might expect, because
            # REAL shapes here score below 0.90: a hard internal colour
            # boundary looks exactly like a shape edge to a texture detector,
            # so it takes a notch out of the shape. The measured 5th percentile
            # on real footage was 0.75.
            #
            # This check runs AFTER the split, not before. Two touching shapes
            # score about 0.87 as one merged blob, so checking first would
            # delete them before they could be separated.
            hull_area = cv2.contourArea(cv2.convexHull(p))
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            # Find the centre using image moments. m00 is the area, m10 and m01
            # are the sums of the x and y coordinates, so dividing gives the
            # average position, i.e. the centroid. This beats "the middle of
            # the bounding box", which is wrong for triangles and other
            # non-symmetric shapes.
            M = cv2.moments(p)
            if M["m00"] == 0:                  # degenerate, avoid dividing by 0
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Is the shape running off the edge of the picture? If so its
            # centroid is the centroid of the VISIBLE PART, not of the whole
            # shape, so the geometry cannot be trusted. We keep it and flag it
            # rather than deleting it, because it is still a real detection.
            x, y, w, h = cv2.boundingRect(p)
            clipped = (x <= border_margin or y <= border_margin
                       or x + w >= W - border_margin or y + h >= H - border_margin)

            detections.append({
                "contour": p,
                "center": (float(cx), float(cy)),
                "xyz": None,                   # estimate_xyz() fills this in
                "area": float(area),
                "solidity": float(solidity),
                "clipped": bool(clipped),
                "was_split": bool(was_split),
            })

    # Biggest first, purely so the output is in a predictable order.
    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections


# =============================================================================
# FINDING THE CIRCLE, WHICH IS OUR ONLY REAL-WORLD RULER
# =============================================================================

def _circularity(contour):
    """How circle-like is this outline? 1.0 is a perfect circle, less otherwise.

    The formula 4*pi*Area / Perimeter^2 comes from the fact that of all shapes
    with a given perimeter, the circle encloses the most area.
    """
    P = cv2.arcLength(contour, True)           # True = the outline is closed
    if P <= 0:
        return 0.0
    return 4.0 * math.pi * cv2.contourArea(contour) / (P * P)


def _ellipse_roundness(contour):
    """Fit an ellipse to the outline and report how well it matches.

    Returns (fill, axis_ratio, semi_major_in_pixels) where:
        fill       = outline area / fitted ellipse area. Near 1.0 if the shape
                     really is an ellipse; lower for a polygon or a merged blob.
        axis_ratio = short axis / long axis. 1.0 for a circle, small for a
                     squashed cigar shape.

    We rely on these more than on _circularity because they are AREA based.
    Circularity uses the perimeter, and our outlines are traced off a pixel mask
    so they are jagged like a staircase, which inflates the measured perimeter.
    Since circularity divides by perimeter SQUARED, that error is doubled: a
    genuinely round circle in this footage scores only 0.79-0.89, not 1.0.
    """
    # fitEllipse returns ((centre_x, centre_y), (width, height), rotation).
    (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)

    # Area of an ellipse is pi * a * b where a and b are the SEMI-axes, so we
    # divide the full widths by 2 each, hence the /4.
    ellipse_area = math.pi * axis_a * axis_b / 4.0
    if ellipse_area <= 0:
        return 0.0, 0.0, 0.0

    return (cv2.contourArea(contour) / ellipse_area,
            min(axis_a, axis_b) / max(axis_a, axis_b),
            max(axis_a, axis_b) / 2.0)         # semi-major = half the long axis


def _find_circle(detections, min_fill=0.96, min_axis_ratio=0.88):
    """Pick out the circle, which is the object whose true size we know.

    Returns (the detection, its radius in pixels), or (None, None).

    Deliberately strict: this one measurement sets the scale for EVERY shape in
    the frame, so it is far better to report nothing than to pick the wrong
    object and silently corrupt every distance.
    """
    best, best_r, best_score = None, None, -1.0

    for d in detections:
        # A shape running off the frame has a truncated outline, and a shape
        # produced by a split has an artificial straight cut across it. Neither
        # has a trustworthy radius.
        if d.get("clipped") or d.get("was_split"):
            continue

        c = d["contour"]
        if len(c) < 5:                         # fitEllipse needs 5+ points
            continue

        # Does an ellipse actually fit this outline, and is that ellipse close
        # to circular rather than a long thin oval?
        fill, axis_ratio, semi_major = _ellipse_roundness(c)
        if fill < min_fill or axis_ratio < min_axis_ratio:
            continue

        # Among everything that qualifies, take the best ellipse fit.
        if fill > best_score:
            best, best_r, best_score = d, semi_major, fill

    return best, best_r


# =============================================================================
# STEP 3 OF THE PIPELINE: PIXELS -> INCHES
# =============================================================================

def estimate_xyz(detections, image_shape, fx=FX, fy=FY, cx=None, cy=None,
                 circle_radius_in=CIRCLE_RADIUS_IN,
                 radius_bias_px=RADIUS_BIAS_PX, z_plane=None):
    """Work out where each shape is in 3D, in inches, relative to the camera.

    THE GEOMETRY. A pinhole camera makes far-away things look small in exact
    proportion to their distance. An object of real size R at distance Z appears
    r pixels across, where:

        r = f * R / Z          (similar triangles)

    We know R for the circle (10 inches) and we can measure r, so:

        Z = f * R / r

    Once Z is known, we can undo the projection for any pixel to get its real
    sideways position:

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    The challenge tells us to assume all the shapes lie on a flat surface, which
    is precisely what lets us reuse the circle's Z for every other shape.

    AXES: +X is right, +Y is DOWN, +Z is away from the camera. Y pointing down
    is the standard camera convention, but it does surprise people reading the
    numbers, so it is worth stating.

    Pass z_plane to supply the depth from outside (see ZPlaneTracker).
    """

    H, W = image_shape[:2]

    # Fill in the principal point. K says (0, 0), which is not believable, so we
    # use the image centre. This affects X and Y only. Z is untouched, because
    # depth depends solely on the focal length and the measured radius.
    if cx is None:
        cx = PRINCIPAL_POINT[0] if PRINCIPAL_POINT else W / 2.0
    if cy is None:
        cy = PRINCIPAL_POINT[1] if PRINCIPAL_POINT else H / 2.0

    # One focal length from the two. They differ by only 0.2% (non-square
    # pixels), and the radius we measure is along an axis at some arbitrary
    # angle, so neither fx nor fy alone is right. The geometric mean is the
    # natural compromise.
    f = math.sqrt(fx * fy)

    ref = None
    if z_plane is not None:
        # Caller supplied the depth, so skip finding the circle entirely.
        Z = float(z_plane)
    else:
        # Find the circle and derive the depth from its radius.
        ref, r_px = _find_circle(detections)
        if ref is None:
            # No ruler, no distances. Say so rather than guessing.
            for d in detections:
                d["xyz"] = None
            return detections

        r_px -= radius_bias_px                 # apply the calibration offset
        if r_px <= 0:
            for d in detections:
                d["xyz"] = None
            return detections

        # We use the SEMI-MAJOR (longest) axis of the fitted ellipse, not the
        # average of the two axes. If the circle is tilted away from the camera
        # it projects to an ellipse, and the long axis lies along the tilt's
        # rotation axis, where every point stays at the same distance Z. So the
        # long axis still obeys r = f*R/Z exactly, while the short one is
        # foreshortened. Averaging the two would over-estimate Z.
        Z = f * circle_radius_in / r_px

    for d in detections:
        # Every shape is on the same flat surface, so they all share this depth.
        d["z_plane"] = float(Z)

        if d.get("clipped"):
            # The depth is still fine, but the centroid belongs to the visible
            # fragment rather than the whole shape, so X and Y would be wrong.
            # Leaving xyz as None keeps the rule "not None means trustworthy".
            d["xyz"] = None
            continue

        u, v = d["center"]
        d["xyz"] = (float((u - cx) * Z / fx), float((v - cy) * Z / fy), float(Z))

    # Mark which shape acted as the ruler, useful for debugging and display.
    if ref is not None:
        ref["is_scale_reference"] = True

    return detections


def detect_shapes(frame, with_depth=False):
    """Run the whole pipeline on one frame. This is the main entry point.

    Deliberately STATELESS: it remembers nothing between calls. Anything that
    needs to persist across video frames lives in the caller's loop, so that a
    single still image can be tested on its own and so that processing two
    videos in one program cannot contaminate one with the other.
    """
    mask = build_shape_mask(frame)                       # find the shapes
    detections = mask_to_detections(mask)                # describe them
    if with_depth:
        detections = estimate_xyz(detections, frame.shape)   # locate them in 3D
    return detections


# =============================================================================
# KEEPING DEPTH ALIVE ACROSS VIDEO FRAMES
# =============================================================================

class ZPlaneTracker:
    """Remembers the scene depth between frames. Create one per video.

    WHY THIS IS NEEDED. Depth comes from the circle, but the circle is only
    cleanly measurable in about a third of frames: for long stretches it
    physically overlaps another shape and there is no round outline to measure.
    Those gaps reach 20 seconds, far too long to simply keep reusing an old
    number while the camera is moving.

    THE TRICK. All the shapes sit on the same flat surface. So the moment we get
    ONE good circle measurement, we can work out the real size of every other
    shape in view:

        real_area = pixel_area * (Z / f)^2

    From then on any of those shapes can serve as the ruler instead:

        Z = f * sqrt(real_area / pixel_area)

    Usage:
        tracker = ZPlaneTracker()
        for frame in frames:
            dets = detect_shapes(frame)
            z = tracker.update(dets)
            if z is not None:
                estimate_xyz(dets, frame.shape, z_plane=z)

    Known limits: shapes are matched between frames by nearest centre, which is
    wrong if two shapes cross paths; learned sizes inherit the error of the
    measurement that produced them; and nothing here helps a frame in which no
    shape at all is clearly visible.
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

        self.last_z = None                # most recent good depth
        self.stale = 0                    # frames since that measurement

        # True when the latest update() measured the circle for real, False when
        # the depth was inferred or reused. Callers need this to report honestly,
        # because estimate_xyz cannot mark a scale reference when it is simply
        # handed a depth.
        self.last_measured_directly = False

    def _match(self, detections):
        """Work out which shape in this frame is which shape from last frame.

        The simplest possible tracker: each shape claims the nearest previously
        seen shape within match_px, and anything unclaimed becomes a new track.
        At 30 fps with slowly moving shapes this is good enough.
        """
        ids, used = [], set()
        for d in detections:
            cx, cy = d["center"]
            best, best_dist = None, self.match_px

            # Find the closest previous track that nothing else has claimed.
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
            self.tracks[best] = (cx, cy)       # remember where it is now
            ids.append(best)

        return ids

    def update(self, detections):
        """Give me this frame's shapes; I will give you the scene depth (inches)."""

        ids = self._match(detections)
        z, trusted = None, False

        # --- Option 1, best: we can see the circle clearly. ---
        ref, r_px = _find_circle(detections)
        if ref is not None and r_px > 0:
            z, trusted = self.f * self.circle_radius_in / r_px, True

        # --- Option 2: infer the depth from a shape we sized earlier. ---
        if z is None:
            votes = []
            for tid, d in zip(ids, detections):
                # Clipped or split shapes have unreliable areas, so skip them.
                if d.get("clipped") or d.get("was_split"):
                    continue
                area = d.get("area", 0.0)
                if tid in self.sizes and area > 0:
                    cand = self.f * math.sqrt(self.sizes[tid] / area)

                    # Sanity-check the vote against the last known depth. Our
                    # nearest-centre matching DOES occasionally confuse two
                    # shapes, and a confused identity does not give a slightly
                    # wrong depth, it gives a wildly wrong one. A camera cannot
                    # change its distance by 3% in a thirtieth of a second, so
                    # any vote claiming that is a matching error, not motion.
                    if self.last_z is None or \
                            abs(cand - self.last_z) <= self.max_rel_step * self.last_z:
                        votes.append(cand)

            # Take the median of the surviving votes so that one bad shape
            # cannot drag the answer around.
            if votes:
                z = float(np.median(votes))

        self.last_measured_directly = trusted

        # --- Option 3: nothing worked. Reuse the old value for a while. ---
        if z is None:
            self.stale += 1
            return self.last_z if self.stale <= self.max_stale else None

        # --- Learn the real size of everything currently visible. ---
        #
        # ONLY when the circle was measured directly. Learning from an inferred
        # depth would feed the estimate back into itself, and it drifts badly:
        # doing that gave depths from 144 to 334 inches on footage whose true
        # range was 236 to 248.
        if trusted:
            for tid, d in zip(ids, detections):
                if d.get("clipped") or d.get("was_split"):
                    continue
                area = d.get("area", 0.0)
                if area > 0:
                    prev = self.sizes.get(tid)
                    new = area * (z / self.f) ** 2
                    # Blend with the previous estimate rather than replacing it,
                    # so one noisy frame does not throw the stored size off.
                    self.sizes[tid] = new if prev is None else 0.7 * prev + 0.3 * new

        self.last_z, self.stale = z, 0
        return z
