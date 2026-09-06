"""Part 5: bring up the whole system.

    ros2 launch pennair_vision pennair.launch.py \
        video:=/abs/path/to/PennAir_2024_App_Dynamic_Hard.mp4

Add rviz:=true for the 3D view, or run rqt_image_view on
/shape_detector/image_annotated for the 2D one.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    video = LaunchConfiguration("video")
    loop = LaunchConfiguration("loop")
    rate = LaunchConfiguration("rate")
    use_rviz = LaunchConfiguration("rviz")

    rviz_cfg = os.path.join(
        get_package_share_directory("pennair_vision"), "rviz", "pennair.rviz")

    source = Node(
        package="pennair_vision",
        executable="video_publisher",
        name="video_publisher",
        output="screen",
        parameters=[{"video_path": video, "loop": loop, "rate": rate,
                     "frame_id": "camera_optical_frame"}],
    )

    detector = Node(
        package="pennair_vision",
        executable="shape_detector",
        name="shape_detector",
        output="screen",
        parameters=[{"publish_markers": True, "publish_annotated": True}],
        # The nodes are wired explicitly rather than by relying on default
        # names, so the graph is readable in `ros2 node info`.
        remappings=[("~/image_raw", "/video_publisher/image_raw"),
                    ("~/camera_info", "/video_publisher/camera_info")],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("video", description="absolute path to the input video"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("rate", default_value="0.0",
                              description="publish Hz; 0 uses the file's own fps"),
        DeclareLaunchArgument("rviz", default_value="false"),
        source, detector, rviz,
    ])
