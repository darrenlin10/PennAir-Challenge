# All shapes found in one frame.

std_msgs/Header header                 # stamp + camera optical frame_id
ShapeDetection[] detections

float64 plane_depth                    # metres; scene depth used for this frame
bool depth_valid                       # false if no depth was available at all
bool depth_measured_directly           # true if the circle was measured this frame,
                                       # false if carried by the plane tracker
