# One detected shape.

# --- image space ------------------------------------------------------------
geometry_msgs/Point center_px          # centroid, pixels (z unused)
geometry_msgs/Point[] outline_px       # contour polygon, pixels (z unused)
float64 area_px
float64 solidity

# --- metric space, camera optical frame (x right, y down, z forward), metres -
bool has_position                      # false when depth was unavailable
geometry_msgs/Point position           # centroid back-projected onto the plane
geometry_msgs/Point[] outline          # contour back-projected onto the plane

bool clipped                           # touches the image border; geometry unreliable
