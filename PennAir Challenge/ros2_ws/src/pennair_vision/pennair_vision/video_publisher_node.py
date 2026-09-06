"""Part 5, node 1: stream a video file out as sensor_msgs/Image.

Also publishes CameraInfo carrying the challenge's K, so the detector gets its
intrinsics over the graph instead of hard-coding them.

    ~/image_raw    sensor_msgs/Image
    ~/camera_info  sensor_msgs/CameraInfo
"""

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from pennair_vision import detector as det


class VideoPublisherNode(Node):

    def __init__(self):
        super().__init__("video_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("loop", True)
        self.declare_parameter("rate", 0.0)      # 0 = use the file's own fps

        path = self.get_parameter("video_path").value
        if not path:
            raise SystemExit("video_publisher: set the video_path parameter")

        # Force the FFMPEG backend. OpenCV built with GStreamer support (which
        # is the norm on a ROS install) treats the filename as a URI, so it
        # stops at the first space and fails on any path containing one --
        # which every file in this challenge does. FFMPEG takes a plain path.
        self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(path)          # fall back to whatever is available
        if not self.cap.isOpened():
            hint = ""
            if not os.path.exists(path):
                hint = "  (that file does not exist)"
            elif " " in path:
                hint = ("  (the path contains spaces and no backend accepted it; "
                        "try renaming the file)")
            raise SystemExit(f"video_publisher: could not open {path}{hint}")

        self.frame_id = self.get_parameter("frame_id").value
        self.loop = bool(self.get_parameter("loop").value)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rate = float(self.get_parameter("rate").value)
        if rate <= 0.0:
            rate = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.bridge = CvBridge()
        self.info = self.build_camera_info()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_image = self.create_publisher(Image, "~/image_raw", qos)
        self.pub_info = self.create_publisher(CameraInfo, "~/camera_info", 10)

        self.n = 0
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"streaming {path} at {rate:.1f} Hz ({self.width}x{self.height}), loop={self.loop}")

    def build_camera_info(self):
        """CameraInfo carrying the challenge's K.

        The principal point is published AS GIVEN, i.e. (0, 0). Inventing
        intrinsics is not this node's job: the detector applies the image
        centre substitution and logs a warning, so the correction shows up in
        the logs instead of being silently baked in at the source.
        """
        info = CameraInfo()
        info.width = self.width
        info.height = self.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [det.FX, 0.0, 0.0,
                  0.0, det.FY, 0.0,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [det.FX, 0.0, 0.0, 0.0,
                  0.0, det.FY, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                self.get_logger().info(f"end of video after {self.n} frames")
                self.timer.cancel()
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                self.timer.cancel()
                return

        stamp = self.get_clock().now().to_msg()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.info.header.stamp = stamp
        self.info.header.frame_id = self.frame_id
        self.pub_image.publish(msg)
        self.pub_info.publish(self.info)
        self.n += 1


def main(args=None):
    rclpy.init(args=args)
    try:
        node = VideoPublisherNode()
    except SystemExit as exc:
        print(exc)
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
