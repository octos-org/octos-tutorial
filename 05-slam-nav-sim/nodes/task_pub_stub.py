#!/usr/bin/env python3
"""Task publisher stub — replaces dora-nav's PostgreSQL-dependent C++ task_pub_node.

Publishes static road attributes (RoadAttri_h) at 10 Hz on tick events.
Values from /home/demo/Public/dora-nav/road_msg.txt.

RoadAttri_h layout (40 bytes = 10 × float32):
  velocity, road_width, aeb_back, aeb_front, aeb_left, aeb_right,
  detect_back, detect_front, detect_left, detect_right
"""

import struct
import pyarrow as pa
from dora import Node


def main():
    node = Node()

    # Static road attributes from road_msg.txt
    # velocity=320, road_width=6, aeb=1.5 all sides, detect=0
    road_attri = struct.pack(
        '<10f',
        320.0,   # velocity
        6.0,     # road_width
        1.5,     # aeb_back
        1.5,     # aeb_front
        1.5,     # aeb_left
        1.5,     # aeb_right
        0.0,     # detect_back
        0.0,     # detect_front
        0.0,     # detect_left
        0.0,     # detect_right
    )
    road_attri_arr = pa.array(list(road_attri), type=pa.uint8())

    print("[task_pub_stub] Publishing road_attri_msg (40 bytes) on each tick")

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            node.send_output("road_attri_msg", road_attri_arr)
        elif event["type"] == "STOP":
            break

    print("[task_pub_stub] Stopped")


if __name__ == "__main__":
    main()
