#!/usr/bin/env python3
"""lon-control — Python port of dora-nav/control/vehicle_control/lon_controller

Simple longitudinal controller: maps Request commands to torque/brake outputs.
NOT a PID — directly passes speed as torque percentage.

Input format:
  Request: Request_h (13 bytes min)
    uint8  reques_type  (0=FORWARD, 1=BACK, 2=STOP, 3=AEB)
    float  run_speed
    float  stop_distance
    float  aeb_distance

Output format:
  TrqBreCmd: TrqBreCmd_h (24 bytes)
    [0]  uint8   trq_enable
    [8]  float32 trq_value_3  (at offset 8 due to alignment)
    [12] uint8   bre_enable
    [16] float32 bre_value    (at offset 16)
    [20] uint8   ACC_DecToStop
    [21] uint8   ACC_Driveoff
"""

import struct

import pyarrow as pa
from dora import Node


# Request types
FORWARD = 0
BACK = 1
STOP = 2
AEB = 3


def pack_trqbrecmd(trq_enable, trq_value, bre_enable, bre_value,
                   acc_dec=0, acc_drive=0):
    """Pack TrqBreCmd_h matching the C++ struct layout.

    The C++ struct has natural alignment:
      offset 0:  uint8_t  trq_enable
      offset 4:  uint32_t trq_value_2  (unused, kept for compatibility)
      offset 8:  float    trq_value_3
      offset 12: uint8_t  bre_enable
      offset 16: float    bre_value    (aligned to 4)
      offset 20: uint8_t  ACC_DecToStop
      offset 21: uint8_t  ACC_Driveoff
    Total: ~24 bytes with padding.
    """
    buf = bytearray(24)
    buf[0] = trq_enable & 0xFF
    struct.pack_into('<f', buf, 8, trq_value)
    buf[12] = bre_enable & 0xFF
    struct.pack_into('<f', buf, 16, bre_value)
    buf[20] = acc_dec & 0xFF
    buf[21] = acc_drive & 0xFF
    return bytes(buf)


def main():
    node = Node()
    print("[lon-control] Longitudinal controller ready")

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] != "Request":
            continue

        raw = bytes(event["value"].to_pylist())
        if len(raw) < 5:
            continue

        req_type = raw[0]
        run_speed = struct.unpack_from('<f', raw, 1)[0]

        if req_type == FORWARD:
            cmd = pack_trqbrecmd(1, run_speed, 0, 0.0)
        elif req_type == BACK:
            cmd = pack_trqbrecmd(1, -run_speed, 0, 0.0)
        elif req_type == STOP:
            cmd = pack_trqbrecmd(0, 0.0, 1, 100.0, acc_dec=1)
        elif req_type == AEB:
            cmd = pack_trqbrecmd(0, 0.0, 1, 300.0)
        else:
            continue

        node.send_output(
            "TrqBreCmd",
            pa.array(list(cmd), type=pa.uint8()),
        )

    print("[lon-control] Stopped")


if __name__ == "__main__":
    main()
