# -*- coding: utf-8 -*-
import time
from leap_hand_tracker import LeapHandTracker
import Leap

def main():
    # Only track the RIGHT hand; use raw palm position (not stabilized)
    tracker = LeapHandTracker(hand_preference='right', use_stabilized=False)

    # Calibrate origins at start (optional)
    print("Place right hand in neutral pose and hold still…")
    time.sleep(1.0)
    ok_pos = tracker.set_origin_xyz()
    ok_ang = tracker.set_origin_rpy()
    print("Calib xyz=%s, rpy=%s" % (ok_pos, ok_ang))

    print("Streaming… (Ctrl+C to quit)")
    while True:
        xyz = tracker.get_xyz()
        rpy = tracker.get_rpy()

        if xyz and rpy:
            # Palm pose (offset by the origins)
            print("Palm xyz(mm): (%.1f, %.1f, %.1f) | rpy(deg): (%.1f, %.1f, %.1f)" %
                  (xyz[0], xyz[1], xyz[2],
                   tracker.rad2deg(rpy[0]), tracker.rad2deg(rpy[1]), tracker.rad2deg(rpy[2])))

            # Index curls & abduction
            idx = tracker.get_finger_curls(Leap.Finger.TYPE_INDEX)
            abd = tracker.get_finger_abduction(Leap.Finger.TYPE_INDEX)
            if idx is not None and abd is not None:
                print("Index curls (deg): MCP=%.1f  PIP=%.1f  DIP=%.1f | Abd(deg)=%.1f" %
                      (tracker.rad2deg(idx['mcp']),
                       tracker.rad2deg(idx['pip']),
                       tracker.rad2deg(idx['dip']),
                       tracker.rad2deg(abd)))

            # Thumb curls & abduction
            th = tracker.get_thumb_curls()
            th_abd = tracker.get_thumb_abduction()
            if th is not None and th_abd is not None:
                print("Thumb curls (deg): MCP=%.1f  IP=%.1f | Abd(deg)=%.1f" %
                      (tracker.rad2deg(th['mcp']),
                       tracker.rad2deg(th['ip']),
                       tracker.rad2deg(th_abd)))

        time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
