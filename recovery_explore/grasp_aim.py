"""Grasp aiming from the wrist camera's depth image: where the object is, and where to put the
open hand so the policy's close lands on it.

Everything here is geometry on the robot's own sensor (an RGB-D wrist camera) and its own pose:
no simulator object state, no segmentation ids. The steps:

* ``cloud_from_depth``      the depth image (metres, top-down rows) and the camera's pixel->world
                            matrix give a world point per pixel, on a stride grid.
* ``fit_support_plane``     RANSAC over near-horizontal planes (normal within PLANE_TILT_DEG of
                            the world vertical): the counter, shelf or appliance floor the object
                            stands on. With a reference point (the last empty closure) the fit
                            uses only the points around it (LOCAL_R_M), so a microwave turntable
                            or a shelf wins over the appliance floor, and the kitchen floor seen
                            far below never does.
* ``segment_blobs``         points more than FOREGROUND_M above the plane are foreground; they
                            are grouped by pixel adjacency (a "blob" per standing thing) and each
                            blob gets its centre, top, footprint axes (PCA in the plane) and size.
* ``choose_blob``           by a hint pixel, else the nearest to a reference point (the last
                            empty closure), else the nearest to the fingertips.
* ``aim_pose``              the grasp point: x, y at the blob's centre; z per mode ("mid": half
                            the blob's height above the plane, clamped to Z_MIN_ABOVE_PLANE_M;
                            "top": Z_TOP_BELOW_M below the top); the wrist yaw that puts the
                            finger axis across the blob's short side; and whether the short side
                            fits between the open fingers.

Units are metres and radians. Rows/cols are top-down image coordinates (row 0 = top), as
``env_service`` renders them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

STRIDE = 4                    # 512 -> 128 pixels a side: 16k points, enough for a shelf and a mug
PLANE_TILT_DEG = 5.0          # a support plane's normal is this close to the world vertical: counters,
                              # shelves and turntables are level, and a looser bound lets RANSAC lay a
                              # tilted plane across a 2 cm step (floor and turntable at once)
PLANE_INLIER_M = 0.006        # RANSAC inlier band
PLANE_ITERS = 200
PLANE_MIN_FRACTION = 0.08     # the plane must hold at least this share of the valid points
UPPER_PLANE_FRACTION = 0.25   # locally, a horizontal plane above the biggest one wins when it holds
UPPER_PLANE_MAX_M = 0.05      #   this share of the local points and sits at most this far above it
                              #   (a turntable or a tray the object stands on, over the floor under it)
FOREGROUND_M = 0.006          # a point this far above the plane stands on it (a spoon is ~1 cm)
FOREGROUND_MAX_M = 0.25       # ... and this far at most (cabinet walls, doors, the arm itself)
BLOB_MIN_PIXELS = 12          # smaller groups are noise / an edge of something
BLOB_MIN_WIDTH_M = 0.008      # narrower than this is an edge seen edge-on (a wall, a door), not a thing
CORE_FRACTION = 0.4           # a blob is measured from its core, the points above this share of its
                              # top (at least FOREGROUND_M): a turntable, a tray or a rim (1-2 cm)
                              # that touches a 7 cm object in the image drops out of its footprint
                              # and centre, while a 1 cm spoon keeps all of its points
BLOB_MAX_XY_M = 0.30          # wider than this is furniture, not a graspable thing
DEPTH_MAX_M = 0.9             # farther than this the wrist camera sees the room, not the workspace
ROBOT_DEPTH_M = 0.12          # nearer than this to the wrist camera is the robot's own gripper and
                              # fingers (the fingertips are ~11 cm from it). Not more: inside a
                              # cabinet a 15-20 cm bottle's top comes within 15 cm of the camera
LOOK_FROM_M = 0.12
LOCAL_R_M = 0.15              # plane fit: points within this of the reference point, horizontally
LOCAL_BELOW_M = 0.30          # ... and at most this far below it
BLOB_R_M = 0.25               # blobs: only things within this of the reference point, horizontally
BLOB_R_WIDE_M = 0.45          # ... and when nothing is there, anything in view this close (the last
                              # miss may be stale: the object was knocked away, or it is another one)
TOP_CAP_M = 0.02              # the centre is taken from the top cap of the blob (this much below its
                              # top): the side faces seen in perspective would pull it off
Z_MIN_ABOVE_PLANE_M = 0.015   # the grip site never goes lower than this over the plane (the fingertips
                              # reach ~1 cm below the grip site: robosuite's panda_gripper.xml puts
                              # grip_site 0.097 along z and the finger tips at 0.0524 + 0.056)
Z_TOP_BELOW_M = 0.02          # "top": this far below the blob's top
OPEN_WIDTH_M = 0.080          # what the fingers open to (Panda: 2 x 0.04 slide); wider blobs do
                              # not fit
WIDTH_MARGIN_M = 0.004        # ... with this margin each side (RoboCasa bottles measure 5.6-6.8 cm)
FINGER_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])   # the fingers slide along the gripper's local x:
                              # panda_gripper.xml has the finger bodies turned +90 deg about z with
                              # their slide joints on their own y, which is the gripper's -x/+x
# the robot's own fingers in the wrist camera's view: two blocks in the bottom corners, always
# closer to the camera than the workspace; points there nearer than FINGER_DEPTH_M are dropped
FINGER_ROWS_FROM = 0.62       # share of the image height where the finger zone starts
FINGER_COLS_SIDE = 0.34       # share of the image width on each side that the zone covers
FINGER_DEPTH_M = 0.17
HOVER_M = 0.04                # cross above the blob's top by this


@dataclass
class Plane:
    point: np.ndarray         # a point on the plane
    normal: np.ndarray        # unit normal, pointing up (positive world z)
    inliers: int
    total: int

    def height_of(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance of points above the plane (m)."""
        with np.errstate(all="ignore"):
            return (np.asarray(pts, dtype=float) - self.point) @ self.normal

    @property
    def z(self) -> float:
        """The plane's height at its reference point (m); for a level plane, its height."""
        return float(self.point[2])


@dataclass
class Blob:
    label: int
    pixels: int
    centre: np.ndarray        # world xyz of the blob's points' mean
    top: float                # highest point's height above the plane (m)
    bottom: float             # lowest point's height above the plane (m)
    axis_long: np.ndarray     # unit vector in the plane's xy, along the footprint's long side
    length: float             # footprint extent along axis_long (m)
    width: float              # ... across it (m)
    rows: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=int))
    cols: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=int))
    view: str = "wrist"

    def summary(self) -> dict[str, Any]:
        return {"pixels": int(self.pixels), "centre": [round(float(v), 4) for v in self.centre],
                "top_cm": round(100.0 * self.top, 1), "bottom_cm": round(100.0 * self.bottom, 1),
                "length_cm": round(100.0 * self.length, 1), "width_cm": round(100.0 * self.width, 1),
                "view": self.view}


GRIPPER_XY_M = 0.12           # a third-person view: points this close to the hand horizontally and
GRIPPER_BELOW_M = 0.03        # above (grip site - this) are the gripper and the arm, not the scene
FIXED_DEPTH_MAX_M = 2.5       # a fixed camera sees the whole kitchen


def cloud_from_depth(depth: np.ndarray, T_p2w: np.ndarray, stride: int = STRIDE,
                     depth_max: float = DEPTH_MAX_M, *, wrist: bool = True,
                     hand_xyz: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points for every ``stride``-th pixel with a usable depth. Returns (points (N,3), rows,
    cols) in the ORIGINAL image's pixel coordinates. A wrist view drops the robot's own gripper by
    depth; a third-person view (``wrist=False``) drops the box around ``hand_xyz`` instead."""
    d = np.asarray(depth, dtype=float)
    h, w = d.shape
    rr, cc = np.mgrid[0:h:stride, 0:w:stride]
    r = rr.reshape(-1)
    c = cc.reshape(-1)
    z = d[r, c]
    ok = np.isfinite(z) & (z > 1e-4) & (z < depth_max)
    if wrist:
        ok &= z >= ROBOT_DEPTH_M                                 # the gripper and fingers themselves
        finger = ((r >= FINGER_ROWS_FROM * h) & ((c <= FINGER_COLS_SIDE * w) | (c >= (1.0 - FINGER_COLS_SIDE) * w))
                  & (z < FINGER_DEPTH_M))
        ok &= ~finger
    r, c, z = r[ok], c[ok], z[ok]
    homog = np.stack([c * z, r * z, z, np.ones_like(z)], axis=-1)
    with np.errstate(all="ignore"):     # Apple's BLAS raises spurious matmul warnings on finite input
        world = homog @ np.asarray(T_p2w, dtype=float).T
    pts = np.asarray(world[:, :3], dtype=float)
    if not wrist and hand_xyz is not None:
        hx = np.asarray(hand_xyz, dtype=float).reshape(3)
        near_hand = (np.hypot(pts[:, 0] - hx[0], pts[:, 1] - hx[1]) <= GRIPPER_XY_M) & (pts[:, 2] >= hx[2] - GRIPPER_BELOW_M)
        pts, r, c = pts[~near_hand], r[~near_hand], c[~near_hand]
    return pts, r, c


def fit_support_plane(pts: np.ndarray, *, iters: int = PLANE_ITERS, inlier_m: float = PLANE_INLIER_M,
                      tilt_deg: float = PLANE_TILT_DEG, seed: int = 0) -> Plane | None:
    """RANSAC for the largest near-horizontal plane among ``pts``. None when nothing qualifies."""
    with np.errstate(all="ignore"):     # Apple's BLAS raises spurious warnings on finite input
        return _fit_support_plane(pts, iters=iters, inlier_m=inlier_m, tilt_deg=tilt_deg, seed=seed)


def _fit_support_plane(pts, *, iters, inlier_m, tilt_deg, seed):
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if n < 30:
        return None
    rng = np.random.default_rng(seed)
    cos_tilt = math.cos(math.radians(tilt_deg))
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for _ in range(int(iters)):
        i = rng.choice(n, 3, replace=False)
        a, b, c = P[i]
        nrm = np.cross(b - a, c - a)
        nn = float(np.linalg.norm(nrm))
        if nn < 1e-9:
            continue
        nrm /= nn
        if nrm[2] < 0:
            nrm = -nrm
        if nrm[2] < cos_tilt:
            continue
        dist = np.abs((P - a) @ nrm)
        k = int(np.count_nonzero(dist <= inlier_m))
        if best is None or k > best[0]:
            best = (k, a, nrm)
    if best is None or best[0] < PLANE_MIN_FRACTION * n:
        return None
    k, a, nrm = best
    # refine on the inliers: least-squares plane through their mean
    inl = P[np.abs((P - a) @ nrm) <= inlier_m]
    mean = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - mean, full_matrices=False)
    nrm2 = vt[-1]
    if nrm2[2] < 0:
        nrm2 = -nrm2
    if nrm2[2] >= cos_tilt:
        a, nrm = mean, nrm2
        k = int(np.count_nonzero(np.abs((P - a) @ nrm) <= inlier_m))
    return Plane(point=np.asarray(a, dtype=float), normal=np.asarray(nrm, dtype=float), inliers=k, total=n)


def segment_blobs(pts: np.ndarray, rows: np.ndarray, cols: np.ndarray, plane: Plane, *,
                  stride: int = STRIDE, min_pixels: int = BLOB_MIN_PIXELS,
                  ref_xyz: Sequence[float] | None = None, radius: float = BLOB_R_M) -> list[Blob]:
    """Things standing on the plane, grouped by pixel adjacency on the stride grid; with
    ``ref_xyz`` only within ``radius`` of it horizontally."""
    P = np.asarray(pts, dtype=float)
    h = plane.height_of(P)
    fg = (h > FOREGROUND_M) & (h < FOREGROUND_MAX_M)
    if ref_xyz is not None:
        ref = np.asarray(ref_xyz, dtype=float).reshape(3)
        fg &= np.hypot(P[:, 0] - ref[0], P[:, 1] - ref[1]) <= float(radius)
    if not fg.any():
        return []
    r = np.asarray(rows) // stride
    c = np.asarray(cols) // stride
    H, W = int(r.max()) + 1, int(c.max()) + 1
    grid = np.zeros((H, W), dtype=bool)
    grid[r[fg], c[fg]] = True
    labels, nlab = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    lab_of_pt = np.zeros(len(P), dtype=int)
    lab_of_pt[fg] = labels[r[fg], c[fg]]
    # in-plane basis for the footprint axes
    up = plane.normal
    ex = np.cross(up, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(ex) < 1e-6:
        ex = np.cross(up, np.array([1.0, 0.0, 0.0]))
    ex /= np.linalg.norm(ex)
    ey = np.cross(up, ex)
    out: list[Blob] = []
    for lab in range(1, nlab + 1):
        m = lab_of_pt == lab
        k = int(np.count_nonzero(m))
        if k < min_pixels:
            continue
        Q_all = P[m]
        hq_all = h[m]
        core = hq_all >= max(FOREGROUND_M, CORE_FRACTION * float(hq_all.max()))
        if np.count_nonzero(core) >= min_pixels:
            Q, hq = Q_all[core], hq_all[core]
        else:
            Q, hq = Q_all, hq_all
        with np.errstate(all="ignore"):
            uv = np.stack([(Q - plane.point) @ ex, (Q - plane.point) @ ey], axis=1)
        mean = uv.mean(axis=0)
        cov = np.cov((uv - mean).T) if k > 2 else np.eye(2) * 1e-8
        vals, vecs = np.linalg.eigh(cov)
        long_uv = vecs[:, int(np.argmax(vals))]
        short_uv = np.array([-long_uv[1], long_uv[0]])
        with np.errstate(all="ignore"):     # Apple's BLAS again (finite input)
            proj_l = (uv - mean) @ long_uv
            proj_s = (uv - mean) @ short_uv
        length = float(proj_l.max() - proj_l.min())
        width = float(proj_s.max() - proj_s.min())
        if max(length, width) > BLOB_MAX_XY_M or width < BLOB_MIN_WIDTH_M:
            continue
        axis_long = long_uv[0] * ex + long_uv[1] * ey
        axis_long /= max(np.linalg.norm(axis_long), 1e-9)
        cap = Q[hq >= hq.max() - TOP_CAP_M]          # the top face: its centre is the footprint centre
        centre = cap.mean(axis=0) if len(cap) >= 3 else Q.mean(axis=0)
        centre = np.array([centre[0], centre[1], Q.mean(axis=0)[2]])
        out.append(Blob(label=lab, pixels=int(len(Q)), centre=centre, top=float(hq.max()), bottom=float(hq_all.min()),
                        axis_long=axis_long, length=length, width=width,
                        rows=np.asarray(rows)[m], cols=np.asarray(cols)[m]))
    out.sort(key=lambda b: -b.pixels)
    return out


def fits_fingers(blob: Blob) -> bool:
    return blob.width + 2 * WIDTH_MARGIN_M <= OPEN_WIDTH_M + 1e-6


NEAR_REF_M = 0.12             # choose_blob: a fitting blob this close to the reference point is
                              # preferred over a nearer one that does not fit; farther ones are not


def choose_blob(blobs: Sequence[Blob], *, hint_pixel: Sequence[int] | None = None,
                ref_xyz: Sequence[float] | None = None, stride: int = STRIDE) -> tuple[Blob | None, str]:
    """The blob under (or nearest, in the image) a hint pixel; else the nearest to ``ref_xyz``
    horizontally among those that fit between the fingers (a plate or a wall next to the object
    is not what a grasp is for), else the nearest of all; else the biggest. Returns (blob, how)."""
    if not blobs:
        return None, "none"
    if hint_pixel is not None:
        hr, hc = int(hint_pixel[0]), int(hint_pixel[1])
        best, bd = None, None
        for b in blobs:
            d = np.hypot(b.rows - hr, b.cols - hc).min()
            if bd is None or d < bd:
                best, bd = b, float(d)
        return best, ("hint" if bd is not None and bd <= 2 * stride else "hint_nearest")
    if ref_xyz is not None:
        ref = np.asarray(ref_xyz, dtype=float).reshape(3)
        dist = lambda x: float(np.hypot(*(x.centre[:2] - ref[:2])))
        nearest = min(blobs, key=dist)
        if fits_fingers(nearest):
            return nearest, "nearest_ref"
        near_fit = [b for b in blobs if fits_fingers(b) and dist(b) <= NEAR_REF_M]
        if near_fit:
            return min(near_fit, key=dist), "nearest_ref_fits"
        return nearest, "nearest_ref"
    pool = [b for b in blobs if fits_fingers(b)] or list(blobs)
    return pool[0], "biggest"


def yaw_to_align(eef_quat: Sequence[float], axis_world: np.ndarray, *, finger_axis_local=FINGER_AXIS_LOCAL) -> float:
    """The turn about the world vertical (rad, in (-pi/2, pi/2]) that brings the gripper's finger
    axis onto ``axis_world`` in the horizontal plane; the gripper is symmetric under half turns."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(np.asarray(eef_quat, dtype=float)).as_matrix()
    f = R @ np.asarray(finger_axis_local, dtype=float)
    a_f = math.atan2(f[1], f[0])
    a_t = math.atan2(axis_world[1], axis_world[0])
    d = (a_t - a_f + math.pi / 2) % math.pi - math.pi / 2
    return float(d)


def aim_pose(blob: Blob, plane: Plane, eef_quat: Sequence[float], *, z_mode: Any = "mid",
             hover: float = HOVER_M) -> dict[str, Any]:
    """The grasp point, the wrist turn and the hover point for ``blob``. ``fits`` says whether the
    short side goes between the open fingers."""
    up = plane.normal
    if isinstance(z_mode, (int, float)) and not isinstance(z_mode, bool):
        z_above = float(z_mode)
    elif str(z_mode) == "top":
        z_above = blob.top - Z_TOP_BELOW_M
    else:
        z_above = 0.5 * blob.top
    z_above = max(Z_MIN_ABOVE_PLANE_M, min(z_above, max(Z_MIN_ABOVE_PLANE_M, blob.top - 0.005)))
    # the blob's centre projected onto the plane, then lifted along the normal
    base = blob.centre - up * float(plane.height_of(blob.centre[None])[0])
    grasp = base + up * z_above
    hover_pt = base + up * (blob.top + hover)
    short_axis = np.cross(up, blob.axis_long)
    short_axis /= max(np.linalg.norm(short_axis), 1e-9)
    yaw = yaw_to_align(eef_quat, short_axis)
    fits = fits_fingers(blob)
    return {"grasp": np.asarray(grasp, dtype=float), "hover": np.asarray(hover_pt, dtype=float),
            "yaw": float(yaw), "fits": bool(fits), "z_above_plane": float(z_above)}


def analyse(depth: np.ndarray, T_p2w: np.ndarray, *, stride: int = STRIDE,
            ref_xyz: Sequence[float] | None = None, wrist: bool = True,
            hand_xyz: Sequence[float] | None = None) -> dict[str, Any]:
    """Plane and blobs from one depth image; the pieces the service's aim op reports. With
    ``ref_xyz`` (the last empty closure, or the fingertips) the plane is fitted around it and
    only the things near it count."""
    pts, rows, cols = cloud_from_depth(depth, T_p2w, stride=stride, wrist=wrist, hand_xyz=hand_xyz,
                                       depth_max=DEPTH_MAX_M if wrist else FIXED_DEPTH_MAX_M)
    plane = None
    local = False
    if len(pts) and ref_xyz is not None:
        ref = np.asarray(ref_xyz, dtype=float).reshape(3)
        near = ((np.hypot(pts[:, 0] - ref[0], pts[:, 1] - ref[1]) <= LOCAL_R_M)
                & (pts[:, 2] <= ref[2] + 0.05) & (pts[:, 2] >= ref[2] - LOCAL_BELOW_M))
        if np.count_nonzero(near) >= 30:
            P_near = pts[near]
            plane = fit_support_plane(P_near)
            local = plane is not None
            if plane is not None:
                # the object stands on the top-most large surface: a turntable or tray over a floor
                h = plane.height_of(P_near)
                above = P_near[(h > PLANE_INLIER_M) & (h <= UPPER_PLANE_MAX_M)]
                if len(above) >= max(30, UPPER_PLANE_FRACTION * len(P_near)):
                    upper = fit_support_plane(above, seed=1)
                    if upper is not None and upper.inliers >= UPPER_PLANE_FRACTION * len(P_near) \
                            and 0.0 < float(plane.height_of(upper.point[None])[0]) <= UPPER_PLANE_MAX_M:
                        plane = upper
    if plane is None and len(pts):
        plane = fit_support_plane(pts)
    blobs = (segment_blobs(pts, rows, cols, plane, stride=stride, ref_xyz=ref_xyz)
             if plane is not None else [])
    wide = False
    if plane is not None and not blobs and ref_xyz is not None:
        blobs = segment_blobs(pts, rows, cols, plane, stride=stride, ref_xyz=ref_xyz, radius=BLOB_R_WIDE_M)
        wide = bool(blobs)
    return {"points": int(len(pts)), "plane": plane, "blobs": blobs, "local_plane": local, "wide_search": wide}


MERGE_M = 0.04                # blobs from two views closer than this are one thing


def analyse_views(views: Sequence[tuple[str, np.ndarray, np.ndarray]], *, ref_xyz: Sequence[float] | None,
                  hand_xyz: Sequence[float] | None, stride: int = STRIDE) -> dict[str, Any]:
    """The scene from several cameras: ``views`` are (name, depth, T_p2w) with the wrist camera
    named "wrist". Each view is analysed on its own (blobs group by pixel adjacency, so clouds are
    not merged); the plane is the wrist view's local plane when it has one, else the first local
    plane found; blobs from all views are pooled, and two within MERGE_M of each other are one
    thing (the one seen with more pixels stays). Each blob carries its ``view``."""
    per: dict[str, dict[str, Any]] = {}
    for name, depth, T in views:
        try:
            per[name] = analyse(depth, T, stride=stride, ref_xyz=ref_xyz, wrist=(name == "wrist"), hand_xyz=hand_xyz)
        except Exception:
            continue
    plane, plane_view = None, None
    for name in ["wrist"] + [n for n in per if n != "wrist"]:
        if name in per and per[name]["plane"] is not None and per[name].get("local_plane"):
            plane, plane_view = per[name]["plane"], name
            break
    if plane is None:
        for name, out in per.items():
            if out["plane"] is not None:
                plane, plane_view = out["plane"], name
                break
    pool: list[Blob] = []
    for name in ["wrist"] + [n for n in per if n != "wrist"]:
        for b in per.get(name, {}).get("blobs", []):
            b.view = name  # type: ignore[attr-defined]
            if plane is not None and per[name]["plane"] is not plane:
                # heights relative to the chosen plane, so views agree; a thing that does not stand
                # on the chosen plane (the kitchen floor seen by a fixed camera under a counter's
                # plane, a higher shelf) is not a candidate
                off = float(plane.height_of(b.centre[None])[0]) - float(per[name]["plane"].height_of(b.centre[None])[0])
                b.top, b.bottom = b.top + off, b.bottom + off
                if b.bottom < -0.02 or b.top < FOREGROUND_M or b.top > FOREGROUND_MAX_M:
                    continue
            dup = next((q for q in pool if float(np.hypot(*(q.centre[:2] - b.centre[:2]))) <= MERGE_M), None)
            if dup is None:
                pool.append(b)
            elif b.pixels > dup.pixels:
                pool[pool.index(dup)] = b
    pool.sort(key=lambda b: -b.pixels)
    return {"plane": plane, "plane_view": plane_view, "blobs": pool,
            "views": {n: {"points": o["points"], "blobs": len(o["blobs"]), "local_plane": bool(o.get("local_plane"))} for n, o in per.items()}}


__all__ = ["Plane", "Blob", "cloud_from_depth", "fit_support_plane", "segment_blobs", "choose_blob", "fits_fingers", "analyse_views",
           "yaw_to_align", "aim_pose", "analyse", "STRIDE", "HOVER_M", "OPEN_WIDTH_M", "FOREGROUND_M",
           "LOOK_FROM_M", "ROBOT_DEPTH_M"]
