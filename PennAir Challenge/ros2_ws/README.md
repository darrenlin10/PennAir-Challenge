# Part 5 — ROS 2

Two packages:

- `pennair_msgs` — `ShapeDetection` / `ShapeDetectionArray`, carrying positions
  **and** outlines, in both pixels and metres.
- `pennair_vision` — the two nodes plus the launch file. `detector.py` is the
  same module `run.py` uses; there is no second copy of the algorithm.

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash          # or humble
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash                 # required, and again in every new shell
```

Build `pennair_msgs` first if you build packages individually — `pennair_vision`
imports the generated Python messages at runtime.

## Run

```bash
ros2 launch pennair_vision pennair.launch.py \
    video:=$HOME/pennair/assets/PennAir_2024_App_Dynamic_Hard.mp4
```

Add `rviz:=true` for the 3D outlines, or view the 2D overlay with:

```bash
ros2 run rqt_image_view rqt_image_view /shape_detector/image_annotated
ros2 topic echo /shape_detector/detections --field detections[0].position
ros2 topic hz /shape_detector/detections
```

## Graph

```
video_publisher ──/video_publisher/image_raw────► shape_detector ──/shape_detector/detections
                └─/video_publisher/camera_info──►                ├─/shape_detector/image_annotated
                                                                 └─/shape_detector/markers
```

## Notes

**Frames.** All metric output is in the camera *optical* frame: x right, y down,
z forward (REP 103 / REP 145). `estimate_xyz` already uses that convention, so
nothing is flipped in the node. To get a body frame (x forward, y left, z up),
publish a static transform and let tf2 do it rather than negating axes by hand.

**Units.** The detector works in inches because the challenge specifies the
circle radius in inches. ROS is metres (REP 103). The conversion happens once,
in the node, at `IN_TO_M`.

**Principal point.** `video_publisher` publishes K exactly as given, principal
point (0, 0) included. `shape_detector` substitutes the image centre and logs a
warning, so the correction is visible in the logs rather than silently baked in
at the source. It affects X and Y only; Z depends solely on fx, fy and the
measured circle radius.

**Dropped frames.** The image topics use BEST_EFFORT with depth 1. The detector
runs well under 30 FPS at 1080p, so frames are dropped by design. A RELIABLE
queue would instead accumulate unbounded latency and report shapes that had
already moved. `ros2 topic hz` on the two image topics shows the gap.

**3D outlines.** Because every shape shares one plane depth, the whole contour
back-projects at that depth, so `outline` is a real metric polygon rather than a
pixel polygon. `outline_stride` (default 4) thins the contour before publishing;
raise it if the topic bandwidth matters.

**Clipped shapes.** A shape touching the image border reports `clipped=true`,
keeps its pixel outline, and has `has_position=false` — its centroid is the
centroid of the visible part, not of the shape.
