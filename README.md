# PennAiR 2026 Software Challenge

## Running it

```bash
pip install opencv-python numpy
python run.py --input "assets/PennAir 2024 App Static.png"       --output out/static.png
python run.py --input "assets/PennAir 2024 App Dynamic.mp4"      --output out/dynamic.mp4
python run.py --input "assets/PennAir 2024 App Dynamic Hard.mp4" --output out/dynamic-hard.mp4
```

Regenerate every tuned constant from a video:

```bash
python measure_constants.py "assets/PennAir 2024 App Dynamic Hard.mp4" --sensitivity
```

## Approach

The background is textured and the shapes are smooth, so the mask is built from
local variance, not color. For each pixel we ask how much the pixels around
it disagree with each other: low inside a shape, high on grass or asphalt. It is
computed from two box filters using `Var = E[I²] − E[I]²`, which is O(1) per
pixel regardless of window size.

Nothing in the pipeline depends on the background being any particular color.

Initially used color to differentiate, but that didn't go well reaching part 3.

## Results

| clip | mean FPS |

| dynamic | 21.6 |

| dynamic hard | 20.8 |

## Challenges

- **Otsu thresholding fails here.** It assumes two classes of comparable size.
  The shapes are ~5% of pixels while the asphalt spans a wide texture range, so
  Otsu finds it cheaper to split the *background* in half than to separate the
  shapes from it. It picks 112 and puts 59% of the frame in the mask. Replaced
  with a fixed fraction of the frame's median texture, which works because the
  background dominates by area, so its median describes the background.

- **Shapes with internal shading get chopped up.** A trapezoid has partly blended corners.
  Raising the cutoff cannot fix it because the populations overlap: shape
  texture is 56 and background is 57. Solved with hysteresis — a loose
  second threshold, kept only where it touches a strict-threshold seed.

- **A static patch of asphalt was detected in every frame.** One spot at
  bbox (1891, 332, 29, 75) whose gravel is slightly smoother than average
  (std 21.5 vs 23.0 nearby). Detection areas turn out sharply bimodal —
  background artifacts under 2000 px, real shapes above 20000 — so `min_area`
  separates them cleanly.

## Part 4 — 3D coordinates

The circle is the only object of known size, so it sets the scene's scale:

```
f = sqrt(fx·fy) = 2567.01 px
Z = f · R / r_px          R = 10 in, r_px = ellipse semi-major axis
X = (u − cx)·Z/fx    Y = (v − cy)·Z/fy
```

Axes are +X right, +Y down, +Z forward — the standard camera convention.

`K`'s principal point is (0, 0), which would put the optical axis at the sensor
corner. Substituted the image centre. This changes X and Y only; Z depends
solely on the focal length and the measured radius.

Semi-major axis, not the mean of the axes: under tilt the major axis lies along
the rotation axis where points stay at depth Z, so `f·R/Z` stays exact.

The circle is cleanly measurable in only ~31% of frames — it spends long
stretches overlapping the triangle. `ZPlaneTracker` uses the flat-surface
assumption to cover the gaps: one good measurement fixes every other shape's
real size, after which any of them can act as the reference. Depth is then
available on 100% of frames.

## Known limitations

- The 10-inch radius is taken from the spec and cannot be verified from the
  video. **Every depth scales linearly with it.**
- `RADIUS_BIAS_PX` is 0.0 — the offset between the mask boundary and the
  circle's true edge is uncalibrated.
- Splitting two touching shapes works to ~27% overlap. Past that a single frame
  doesn't contain the information; it needs tracking across frames.
- Constants were measured on the dynamic-hard clip, whose background is a static
  image with shapes composited over it. **[Re-run measure_constants.py on the
  other clips and say whether the values hold.]**

## Attribution

Used Claude Opus 5 for help
