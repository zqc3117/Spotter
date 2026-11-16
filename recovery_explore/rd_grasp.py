"""Depth-based object segmentation and top-down grasp planning for the RoboDojo judge (ARX X5, parallel gripper).

Only sensor data a real robot has: RGB-D frames, camera intrinsics, and each camera's pose (head camera fixed,
wrist cameras through forward kinematics + hand-eye calibration). No simulator object state.

Pipeline, all numpy:
  backproject   depth (metres along the optical axis) -> world points; camera frame is USD (x right, y up,
                looking down -z), checked against the table plane (z = 0.765 on every camera) on 2026-10-21
  support_plane the dominant horizontal surface near the seed (z histogram mode, 5 mm bins)
  segment       points above that surface, away from the robot's own hands, grown as one connected blob on a
                1 cm grid from the cell nearest the seed
  plan_top_down centroid, top height, PCA in the plane -> close the fingers across the short side
"""
from __future__ import annotations

import numpy as np

CELL = 0.01          # grid cell for connectivity (m)
ABOVE = 0.008        # a point belongs to an object when it is this far above the support (m)
SEARCH_R = 0.15      # look for the object within this radius of the seed (m)
HAND_R = 0.07        # drop points this close to a hand (the robot seeing itself)
STEP_Z = 0.02       # max height step between connected grid cells (m)
BONE_R = 0.06        # drop points this close to an arm segment (base -> shoulder -> wrist)


def backproject(depth: np.ndarray, K: np.ndarray, T: np.ndarray, stride: int = 2,
                zmin: float = 0.05, zmax: float = 2.5):
    """depth HxW -> (N,3) world points and their (row, col)."""
    h, w = depth.shape[:2]
    rr, cc = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[rr, cc]
    ok = np.isfinite(z) & (z > zmin) & (z < zmax)
    rr, cc, z = rr[ok], cc[ok], z[ok]
    x = (cc - K[0, 2]) / K[0, 0] * z
    y = (rr - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, -y, -z], axis=1)           # USD camera frame
    pw = pc @ T[:3, :3].T + T[:3, 3]
    return pw, np.stack([rr, cc], axis=1)


def pixel_to_world(depth: np.ndarray, K: np.ndarray, T: np.ndarray, row: float, col: float, win: int = 3):
    """One pixel -> world point; takes the nearest valid depth in a small window (edges of thin objects)."""
    h, w = depth.shape[:2]
    r0, c0 = int(round(row)), int(round(col))
    best = None
    for dr in range(-win, win + 1):
        for dc in range(-win, win + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < h and 0 <= c < w:
                z = float(depth[r, c])
                if np.isfinite(z) and 0.05 < z < 2.5 and (best is None or dr * dr + dc * dc < best[0]):
                    best = (dr * dr + dc * dc, r, c, z)
    if best is None:
        return None
    _, r, c, z = best
    x = (c - K[0, 2]) / K[0, 0] * z
    y = (r - K[1, 2]) / K[1, 1] * z
    return T[:3, :3] @ np.array([x, -y, -z]) + T[:3, 3]


def support_plane(points: np.ndarray, seed_xy: np.ndarray, radius: float = 0.30) -> float | None:
    near = points[np.linalg.norm(points[:, :2] - seed_xy, axis=1) < radius]
    if len(near) < 50:
        return None
    zs = near[:, 2]
    lo, hi = np.percentile(zs, 1), np.percentile(zs, 99)
    bins = np.arange(lo, hi + 0.005, 0.005)
    if len(bins) < 3:
        return float(np.median(zs))
    hist, edges = np.histogram(zs, bins=bins)
    k = int(np.argmax(hist))
    sel = zs[(zs >= edges[k]) & (zs < edges[k + 1])]
    return float(np.median(sel))


def _seg_dist(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a; t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def segment_snap(points, seed_xy, table_z, hands, bones=(), radius2=0.30):
    """segment() at the seed; if nothing is there, the nearest object within radius2 (a pointing miss of a few cm
    should not end the grasp). Returns (blob, snapped_distance_m or 0.0)."""
    blob = segment(points, seed_xy, table_z, hands, bones)
    if blob is not None:
        return blob, 0.0
    d = np.linalg.norm(points[:, :2] - seed_xy, axis=1)
    m = (d < radius2) & (points[:, 2] > table_z + ABOVE) & (points[:, 2] < table_z + 0.40)
    for h in hands:
        m &= np.linalg.norm(points - h, axis=1) > HAND_R
    for a, b in bones:
        m &= _seg_dist(points, a, b) > BONE_R
    if m.sum() < 20:
        return None, 0.0
    near = points[m][np.argmin(d[m])]
    blob = segment(points, near[:2], table_z, hands, bones)
    return blob, (float(np.linalg.norm(near[:2] - seed_xy)) if blob is not None else 0.0)


def segment(points: np.ndarray, seed_xy: np.ndarray, table_z: float, hands: list[np.ndarray], bones=()):
    """Connected blob of points above the support nearest the seed. Returns (N,3) or None."""
    d = np.linalg.norm(points[:, :2] - seed_xy, axis=1)
    m = (d < SEARCH_R) & (points[:, 2] > table_z + ABOVE) & (points[:, 2] < table_z + 0.40)
    for h in hands:
        m &= np.linalg.norm(points - h, axis=1) > HAND_R
    for a, b in bones:
        m &= _seg_dist(points, a, b) > BONE_R
    pts = points[m]
    if len(pts) < 20:
        return None
    cells = np.floor((pts[:, :2] - seed_xy) / CELL).astype(int)
    occ = {}
    for i, c in enumerate(map(tuple, cells)):
        occ.setdefault(c, []).append(i)
    top = {c: float(np.max(pts[idx, 2])) for c, idx in occ.items()}
    start = min(occ, key=lambda c: c[0] * c[0] + c[1] * c[1])
    seen, stack = {start}, [start]
    while stack:
        cx, cy = stack.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (cx + dx, cy + dy)
                # height-aware: neighbouring cells join only without a step of more than STEP_Z, so a flat
                # object lying against a tall one (bread next to the toaster) stays its own blob
                if n in occ and n not in seen and abs(top[n] - top[(cx, cy)]) <= STEP_Z:
                    seen.add(n); stack.append(n)
    idx = [i for c in seen for i in occ[c]]
    blob = pts[idx]
    return blob if len(blob) >= 20 else None


def plan_top_down(blob: np.ndarray, table_z: float):
    """Grasp centre, top height, short-axis direction and widths of a blob."""
    xy = blob[:, :2]
    c = np.median(xy, axis=0)
    cov = np.cov((xy - c).T)
    evals, evecs = np.linalg.eigh(cov)
    minor, major = evecs[:, 0], evecs[:, 1]
    pm, pM = (xy - c) @ minor, (xy - c) @ major
    width = float(np.percentile(pm, 97) - np.percentile(pm, 3))
    length = float(np.percentile(pM, 97) - np.percentile(pM, 3))
    top = float(np.percentile(blob[:, 2], 95))
    height = top - table_z
    return {"center_xy": c, "top_z": top, "height": height, "minor": minor, "width": width, "length": length,
            "n_points": int(len(blob))}


def top_down_quat(close_dir_xy: np.ndarray, prefer_y: np.ndarray | None = None):
    """wxyz quaternion with the gripper pointing down (EE +x = world -z) and the fingers closing along close_dir
    (EE y). Of the two equivalent signs, take the one nearer prefer_y to keep the wrist roll small."""
    y = np.array([close_dir_xy[0], close_dir_xy[1], 0.0]); y /= np.linalg.norm(y) + 1e-12
    if prefer_y is not None and float(np.dot(y, prefer_y)) < 0:
        y = -y
    x = np.array([0.0, 0.0, -1.0]); z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q); return q / np.linalg.norm(q)
