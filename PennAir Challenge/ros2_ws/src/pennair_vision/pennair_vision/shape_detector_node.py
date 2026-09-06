"""Part 5, node 2: run the shape detector on an incoming image stream.

    subscribed
        ~/image_raw        sensor_msgs/Image
        ~/camera_info      sensor_msgs/CameraInfo    optional; overrides built-in K
    published
        ~/detections       pennair_msgs/ShapeDetectionArray   positions + outlines
        ~/image_annotated  sensor_msgs/Image                  contours drawn on
        ~/markers          visualization_msgs/MarkerArray     RViz outlines
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

from pennair_msgs.msg import ShapeDetection, ShapeDetectionArray
from pennair_vision import detector as det
from pennair_vision.detector import ZPlaneTracker, detect_shapes, estimate_xyz


# REP 103 mandates metres. The detector works in inches because the challenge
# specifies the circle radius in inches. Convert once, here at the boundary.
IN_TO_M = 0.0254


def intrinsics_from_k(k, width, height):
    """(fx, fy, cx, cy, substituted) from a row-major 3x3 K.

    A K with cx = cy = 0 is a matrix written down without its principal point,
    not a camera whose optical axis leaves through the top-left corner of the
    sensor. Substituting the image centre changes X and Y only -- Z depends
    solely on fx, fy and the measured radius.
    """
    fx, fy = float(k[0]), float(k[4])
    cx, cy = float(k[2]), float(k[5])
    substituted = abs(cx) < 1.0 and abs(cy) < 1.0
    if substituted:
        cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy, substituted


def backproject(u, v, z_m, fx, fy, cx, cy):
    """Pixel + plane depth -> 3D point in metres, camera optical frame."""
    return ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m)


class ShapeDetectorNode(Node):

    def __init__(self):
        super().__init__("shape_detector")

        self.declare_parameter("publish_markers", True)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("outline_stride", 4)     # thin the polygon for the wire
        self.declare_parameter("marker_lifetime_sec", 0.3)

        self.bridge = CvBridge()
        self.tracker = ZPlaneTracker()
        self.intrinsics = None
        self.n_frames = self.n_depth = self.n_direct = 0
        self.proc_total = 0.0

        # BEST_EFFORT depth 1. The detector runs well under 30 FPS at 1080p, so
        # frames WILL be dropped. Dropping is correct; a RELIABLE queue would
        # build unbounded latency and report shapes that moved seconds ago.
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, "~/image_raw", self.on_image, sensor_qos)
        self.create_subscription(CameraInfo, "~/camera_info", self.on_info, 10)

        self.pub_det = self.create_publisher(ShapeDetectionArray, "~/detections", 10)
        self.pub_img = self.create_publisher(Image, "~/image_annotated", sensor_qos)
        self.pub_mrk = self.create_publisher(MarkerArray, "~/markers", 10)

        self.create_timer(5.0, self.report)
        self.get_logger().info("shape_detector ready")

    def on_info(self, msg):
        if self.intrinsics is not None:
            return                                   # intrinsics are static
        fx, fy, cx, cy, substituted = intrinsics_from_k(msg.k, msg.width, msg.height)
        self.intrinsics = (fx, fy, cx, cy)
        self.get_logger().info(
            f"intrinsics from camera_info: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
        if substituted:
            self.get_logger().warning(
                "camera_info principal point was (0, 0); substituted the image "
                "centre. Affects X and Y only, not Z.")

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:                     # noqa: BLE001
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return

        h, w = frame.shape[:2]
        if self.intrinsics is None:
            self.intrinsics = (det.FX, det.FY, w / 2.0, h / 2.0)
            self.get_logger().warning("no camera_info yet; using K from detector.py")
        fx, fy, cx, cy = self.intrinsics

        t0 = self.get_clock().now()
        detections = detect_shapes(frame)
        z_in = self.tracker.update(detections)
        if z_in is not None:
            estimate_xyz(detections, frame.shape, fx=fx, fy=fy, cx=cx, cy=cy,
                         z_plane=z_in)
        self.proc_total += (self.get_clock().now() - t0).nanoseconds * 1e-9

        self.n_frames += 1
        self.n_depth += int(z_in is not None)
        direct = bool(getattr(self.tracker, "last_measured_directly", False))
        self.n_direct += int(direct)

        self.publish(msg.header, frame, detections, z_in, direct, fx, fy, cx, cy)

    def publish(self, header, frame, detections, z_in, direct, fx, fy, cx, cy):
        stride = max(1, int(self.get_parameter("outline_stride").value))
        z_m = None if z_in is None else z_in * IN_TO_M

        out = ShapeDetectionArray()
        out.header = header
        out.depth_valid = z_m is not None
        out.plane_depth = float(z_m) if z_m is not None else 0.0
        out.depth_measured_directly = direct

        annotated = frame.copy() if self.get_parameter("publish_annotated").value else None

        for d in detections:
            u, v = d["center"]
            pts = d["contour"].reshape(-1, 2)[::stride]

            m = ShapeDetection()
            m.center_px = Point(x=float(u), y=float(v), z=0.0)
            m.outline_px = [Point(x=float(px), y=float(py), z=0.0) for px, py in pts]
            m.area_px = float(d["area"])
            m.solidity = float(d["solidity"])
            m.clipped = bool(d["clipped"])

            # A clipped shape's centroid is the centroid of the visible part,
            # not of the shape, so no metric position is claimed for it -- but
            # the pixel outline is still published, because it is true.
            m.has_position = d.get("xyz") is not None
            if m.has_position:
                X, Y, Z = d["xyz"]
                m.position = Point(x=X * IN_TO_M, y=Y * IN_TO_M, z=Z * IN_TO_M)
                # Every shape lies on the same plane, so the whole outline
                # back-projects at that one depth. This is what makes the
                # outline metric rather than just a pixel polygon.
                m.outline = [Point(x=X2, y=Y2, z=Z2) for X2, Y2, Z2 in
                             (backproject(px, py, Z * IN_TO_M, fx, fy, cx, cy)
                              for px, py in pts)]
            out.detections.append(m)

            if annotated is not None:
                cv2.drawContours(annotated, [d["contour"]], -1, (0, 255, 0), 2)
                cv2.circle(annotated, (int(u), int(v)), 5, (0, 0, 255), -1)
                if m.has_position:
                    cv2.putText(annotated, f"{m.position.z:.2f}m",
                                (int(u) + 8, int(v) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.pub_det.publish(out)

        if annotated is not None:
            img = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img.header = header
            self.pub_img.publish(img)

        if self.get_parameter("publish_markers").value:
            self.pub_mrk.publish(self.make_markers(out))

    def make_markers(self, out):
        arr = MarkerArray()
        life = Duration(seconds=float(
            self.get_parameter("marker_lifetime_sec").value)).to_msg()

        clear = Marker()
        clear.header = out.header
        clear.action = Marker.DELETEALL      # else vanished shapes linger in RViz
        arr.markers.append(clear)

        for i, d in enumerate(out.detections):
            if not d.has_position or not d.outline:
                continue
            line = Marker()
            line.header = out.header
            line.ns = "outlines"
            line.id = i
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.01
            line.color.r, line.color.g, line.color.b, line.color.a = 0.0, 1.0, 0.4, 1.0
            line.points = list(d.outline) + [d.outline[0]]     # close the loop
            line.lifetime = life
            arr.markers.append(line)

            dot = Marker()
            dot.header = out.header
            dot.ns = "centres"
            dot.id = i
            dot.type = Marker.SPHERE
            dot.action = Marker.ADD
            dot.pose.position = d.position
            dot.pose.orientation.w = 1.0
            dot.scale.x = dot.scale.y = dot.scale.z = 0.05
            dot.color.r, dot.color.g, dot.color.b, dot.color.a = 1.0, 0.2, 0.2, 1.0
            dot.lifetime = life
            arr.markers.append(dot)
        return arr

    def report(self):
        if not self.n_frames:
            return
        mean = self.proc_total / self.n_frames
        self.get_logger().info(
            f"{self.n_frames} frames | {1.0 / mean:.1f} FPS ({mean * 1000:.0f} ms) | "
            f"depth {self.n_depth / self.n_frames:.0%} "
            f"({self.n_direct / self.n_frames:.0%} direct from the circle)")


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
