# -*- coding: utf-8 -*-
# Python 2.7-compatible Leap wrapper for single-hand tracking + joint angles

import math
import sys
import time

# Make sure Leap.py (the SWIG wrapper) is importable (adjust as needed)
# e.g., sys.path.insert(0, "/path/to/LeapSDK/lib")
import Leap

RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0
_EPS = 1e-6

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _angle_between(u, v):
    """Unsigned angle (radians) between vectors u and v."""
    # Leap.Vector has dot() and angle_to(), but we clamp for numerical safety.
    dot = u.dot(v)
    mu = u.magnitude
    mv = v.magnitude
    if mu < _EPS or mv < _EPS:
        return 0.0
    c = _clamp(dot / (mu * mv), -1.0, 1.0)
    return math.acos(c)

def _project_onto_plane(v, plane_normal):
    # v_proj = v - n * (v·n)
    n = plane_normal
    return Leap.Vector(v.x - n.x * v.dot(n),
                       v.y - n.y * v.dot(n),
                       v.z - n.z * v.dot(n))

def _signed_angle_in_plane(u, v, plane_normal):
    """
    Signed angle (radians) from v -> u within the plane whose normal is plane_normal.
    Positive when rotation follows right-hand rule about plane_normal.
    """
    u_proj = _project_onto_plane(u, plane_normal)
    v_proj = _project_onto_plane(v, plane_normal)

    # normalize projections
    um = u_proj.magnitude
    vm = v_proj.magnitude
    if um < _EPS or vm < _EPS:
        return 0.0

    uhat = Leap.Vector(u_proj.x / um, u_proj.y / um, u_proj.z / um)
    vhat = Leap.Vector(v_proj.x / vm, v_proj.y / vm, v_proj.z / vm)

    # angle and sign via cross
    ang = _angle_between(uhat, vhat)  # [0, pi]
    cross = Leap.Vector(uhat.y * vhat.z - uhat.z * vhat.y,
                        uhat.z * vhat.x - uhat.x * vhat.z,
                        uhat.x * vhat.y - uhat.y * vhat.x)
    sgn = 1.0 if cross.dot(plane_normal) >= 0.0 else -1.0
    return ang * sgn


class LeapHandTracker(object):
    """
    Single-hand tracker. Exposes:
      - get_xyz(), set_origin_xyz()
      - get_rpy(), set_origin_rpy()
      - get_finger_curls(finger_type) -> dict(mcp, pip, dip)
      - get_finger_abduction(finger_type) -> signed radians (+: right, -: left)
      - thumb getters: get_thumb_curls() -> dict(mcp, ip), get_thumb_abduction()
      - convenience per-joint getters for each finger
    """
    def __init__(self, hand_preference='right', use_stabilized=False):
        """
        :param hand_preference: 'right' or 'left' (we’ll only return that hand)
        :param use_stabilized: if True, xyz origin uses stabilized palm position
        """
        self.controller = Leap.Controller()
        self.hand_pref_right = (str(hand_preference).lower() == 'right')
        self.use_stabilized = bool(use_stabilized)

        # Origins (offsets)
        self._xyz_origin = Leap.Vector(0.0, 0.0, 0.0)
        self._rpy_origin = (0.0, 0.0, 0.0)  # radians

        # Cache last hand for getters that don't want to explicitly pass a frame
        self._last_hand = None

    # ---------- Core frame/hand selection ----------

    def _get_frame(self):
        return self.controller.frame()

    def _select_hand(self, frame=None):
        if frame is None:
            frame = self._get_frame()
        best = None
        for hand in frame.hands:
            if self.hand_pref_right and hand.is_right:
                best = hand
                break
            if (not self.hand_pref_right) and hand.is_left:
                best = hand
                break
        self._last_hand = best
        return best

    # ---------- Pose getters / setters ----------

    def set_origin_xyz(self):
        """Set xyz origin to the current palm position (mm)."""
        hand = self._select_hand()
        if not hand:
            return False
        pos = hand.stabilized_palm_position if self.use_stabilized else hand.palm_position
        self._xyz_origin = Leap.Vector(pos.x, pos.y, pos.z)
        return True

    def set_origin_rpy(self):
        """Set rpy origin (radians) to the current hand’s roll, pitch, yaw."""
        hand = self._select_hand()
        if not hand:
            return False
        r, p, y = self._compute_rpy(hand)
        self._rpy_origin = (r, p, y)
        return True

    def _compute_rpy(self, hand):
        """
        Roll, pitch, yaw (radians) following Leap’s convention used in the sample:
          pitch from hand.direction.pitch
          roll  from hand.palm_normal.roll
          yaw   from hand.direction.yaw
        """
        # NOTE: These properties exist on Leap.Vector via SWIG, in radians.
        pitch = hand.direction.pitch  # around x (right) axis
        roll  = hand.palm_normal.roll # around z (forward) axis (device coords)
        yaw   = hand.direction.yaw    # around y (up) axis
        return (roll, pitch, yaw)

    def get_xyz(self):
        """
        Returns (x, y, z) in millimeters, offset by the xyz origin if set.
        """
        hand = self._select_hand()
        if not hand:
            return None
        pos = hand.stabilized_palm_position if self.use_stabilized else hand.palm_position
        off = self._xyz_origin
        return (pos.x - off.x, pos.y - off.y, pos.z - off.z)

    def get_rpy(self):
        """
        Returns (roll, pitch, yaw) in radians, offset by the rpy origin.
        """
        hand = self._select_hand()
        if not hand:
            return None
        r, p, y = self._compute_rpy(hand)
        r0, p0, y0 = self._rpy_origin
        return (r - r0, p - p0, y - y0)

    # ---------- Finger utilities ----------

    @staticmethod
    def _bones_for_finger(finger):
        """
        Returns the four bones (metacarpal, proximal, intermediate, distal).
        Some thumbs may report zero-length metacarpal or missing intermediate.
        """
        return (finger.bone(Leap.Bone.TYPE_METACARPAL),
                finger.bone(Leap.Bone.TYPE_PROXIMAL),
                finger.bone(Leap.Bone.TYPE_INTERMEDIATE),
                finger.bone(Leap.Bone.TYPE_DISTAL))

    @staticmethod
    def _curl_angle_between(bone_a, bone_b):
        """
        Joint curl angle between two consecutive bones (radians).
        0 when straight; increases as the finger bends (inner angle).
        """
        # Use bone direction vectors.
        u = bone_a.direction
        v = bone_b.direction
        return _angle_between(u, v)

    def _finger_by_type(self, hand, finger_type):
        for f in hand.fingers:
            if f.type == finger_type:
                return f
        return None

    def _palm_basis(self, hand):
        """
        Returns (x_basis, y_basis, z_basis) from Hand.basis.
        Leap’s hand basis:
          x_basis ~ palm right, y_basis ~ palm up, z_basis ~ palm forward
        """
        b = hand.basis
        return (b.x_basis, b.y_basis, b.z_basis)

    # ---------- Abduction (side-to-side spread) ----------

    def get_finger_abduction(self, finger_type):
        """
        Signed abduction angle (radians) for the given finger, relative to palm forward.
        + means towards palm's +x (to the right), − towards -x (left).
        """
        hand = self._select_hand()
        if not hand:
            return None
        finger = self._finger_by_type(hand, finger_type)
        if not finger:
            return None

        x_b, y_b, z_b = self._palm_basis(hand)
        # Direction of the proximal bone defines the finger's base direction.
        _, prox, _, _ = self._bones_for_finger(finger)
        fdir = prox.direction

        # Work in the palm plane (normal = y_b)
        # Reference: forward (z_b). Signed angle in plane from z_b to fdir.
        ang = _signed_angle_in_plane(fdir, z_b, y_b)
        # For signedness toward the right (+x), the right-hand rule with normal=y_b
        # already gives positive when rotating z_b toward +x.
        return ang

    # ---------- Curl getters for non-thumb fingers ----------

    def get_finger_curls(self, finger_type):
        """
        Returns dict with curl angles (radians): {'mcp':..., 'pip':..., 'dip':...}
        for INDEX/MIDDLE/RING/PINKY. For thumb, use get_thumb_curls().
        """
        hand = self._select_hand()
        if not hand:
            return None
        finger = self._finger_by_type(hand, finger_type)
        if not finger:
            return None

        meta, prox, inter, dist = self._bones_for_finger(finger)

        # MCP: between metacarpal and proximal
        mcp = self._curl_angle_between(meta, prox)

        # PIP: between proximal and intermediate
        pip = self._curl_angle_between(prox, inter)

        # DIP: between intermediate and distal
        dip = self._curl_angle_between(inter, dist)

        return {'mcp': mcp, 'pip': pip, 'dip': dip}

    # ---------- Thumb-specific ----------

    def get_thumb_curls(self):
        """
        Returns dict with thumb curl angles (radians):
           {'mcp': ..., 'ip': ...}
        where MCP is between metacarpal and proximal, and IP between proximal and distal.
        Falls back across missing/zero-length bones if necessary.
        """
        hand = self._select_hand()
        if not hand:
            return None
        thumb = self._finger_by_type(hand, Leap.Finger.TYPE_THUMB)
        if not thumb:
            return None

        meta, prox, inter, dist = self._bones_for_finger(thumb)

        # Thumb MCP: metacarpal vs proximal (fallback to proximal vs intermediate/distal)
        if meta.length > _EPS:
            mcp = self._curl_angle_between(meta, prox)
        else:
            # Some SDK builds report near-zero metacarpal; fall back
            if inter.length > _EPS:
                mcp = self._curl_angle_between(prox, inter)
            else:
                mcp = self._curl_angle_between(prox, dist)

        # Thumb IP: ideally between proximal and distal (skip intermediate if zero)
        if inter.length > _EPS:
            ip = self._curl_angle_between(inter, dist)
        else:
            ip = self._curl_angle_between(prox, dist)

        return {'mcp': mcp, 'ip': ip}

    def get_thumb_abduction(self):
        """
        Signed abduction (radians) for the thumb: angle in palm plane (normal=y_b)
        from x_b (palm right) to the thumb proximal direction.
        Positive when thumb moves forward (toward +z) from +x under right-hand rule,
        negative when it moves backward from +x toward -z.
        """
        hand = self._select_hand()
        if not hand:
            return None
        thumb = self._finger_by_type(hand, Leap.Finger.TYPE_THUMB)
        if not thumb:
            return None

        x_b, y_b, z_b = self._palm_basis(hand)
        _, prox, _, dist = self._bones_for_finger(thumb)
        tdir = prox.direction if prox.length > _EPS else dist.direction

        ang = _signed_angle_in_plane(tdir, x_b, y_b)
        return ang

    # ---------- Convenience per-finger getters ----------

    # Index
    def get_index_mcp_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_INDEX)['mcp']
    def get_index_pip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_INDEX)['pip']
    def get_index_dip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_INDEX)['dip']
    def get_index_abduction(self): return self.get_finger_abduction(Leap.Finger.TYPE_INDEX)

    # Middle
    def get_middle_mcp_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_MIDDLE)['mcp']
    def get_middle_pip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_MIDDLE)['pip']
    def get_middle_dip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_MIDDLE)['dip']
    def get_middle_abduction(self): return self.get_finger_abduction(Leap.Finger.TYPE_MIDDLE)

    # Ring
    def get_ring_mcp_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_RING)['mcp']
    def get_ring_pip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_RING)['pip']
    def get_ring_dip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_RING)['dip']
    def get_ring_abduction(self): return self.get_finger_abduction(Leap.Finger.TYPE_RING)

    # Pinky
    def get_pinky_mcp_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_PINKY)['mcp']
    def get_pinky_pip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_PINKY)['pip']
    def get_pinky_dip_curl(self): return self.get_finger_curls(Leap.Finger.TYPE_PINKY)['dip']
    def get_pinky_abduction(self): return self.get_finger_abduction(Leap.Finger.TYPE_PINKY)

    # Thumb
    def get_thumb_mcp_curl(self): return self.get_thumb_curls()['mcp']
    def get_thumb_ip_curl(self):  return self.get_thumb_curls()['ip']

    # ---------- Helpers (degrees, optional) ----------

    @staticmethod
    def rad2deg(r):
        return r * RAD_TO_DEG

    @staticmethod
    def deg2rad(d):
        return d * DEG_TO_RAD


if __name__ == "__main__":
    # Tiny smoke test: print xyz + yaw for the preferred hand for ~2 seconds.
    tracker = LeapHandTracker(hand_preference='right', use_stabilized=False)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        xyz = tracker.get_xyz()
        rpy = tracker.get_rpy()
        if xyz and rpy:
            print("xyz(mm): (%.1f, %.1f, %.1f)  |  yaw(deg): %.1f" %
                  (xyz[0], xyz[1], xyz[2], tracker.rad2deg(rpy[2])))
        time.sleep(0.01)
