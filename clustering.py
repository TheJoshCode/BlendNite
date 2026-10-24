"""
Pure-numpy cluster + LOD builder (no bpy import, safe to run in worker threads).

Per asset (one unique mesh geometry):
  - LOD 0 is the original mesh.  Coarser LODs come from grid vertex clustering at
    halving resolution (~4x fewer triangles per level), each with a known object-space
    geometric error.
  - Every LOD is split into spatially coherent clusters of `cluster_size` triangles by
    sorting triangle centroids along a Morton (Z-order) curve and chunking.  Because a
    coarse LOD has ~4x fewer triangles, its clusters cover ~4x more surface:
    "farther away => bigger clusters".
  - Each cluster stores a bounding sphere and a normal cone for frustum / back-face
    culling.

Everything is vectorised; there are no per-triangle Python loops.
"""
from __future__ import annotations

import math
import os

import numpy as np

FORMAT_VERSION = 3
_FIELDS = ("positions", "normals", "tris", "cluster_start",
           "centers", "radii", "cone_axis", "cone_cutoff", "uvs")


class LODLevel:
    __slots__ = _FIELDS + ("error",)

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def n_tris(self) -> int:
        return int(len(self.tris))

    @property
    def n_clusters(self) -> int:
        return int(len(self.cluster_start) - 1)


class ClusterAsset:
    """Everything needed to draw one unique geometry at any distance.
    Stored in object-local space so any number of objects can share it."""

    def __init__(self, key, levels, center, radius, cluster_size):
        self.key = key
        self.levels = levels
        self.center = np.asarray(center, np.float64)
        self.radius = float(radius)
        self.cluster_size = int(cluster_size)

    @property
    def errors(self) -> np.ndarray:
        return np.array([l.error for l in self.levels], np.float64)

    @property
    def tri_count(self) -> int:
        return self.levels[0].n_tris


# ── helpers ───────────────────────────────────────────────────────────────

def _part1by2(x):
    u = np.uint64
    x = x & u(0x1FFFFF)
    x = (x | (x << u(32))) & u(0x1F00000000FFFF)
    x = (x | (x << u(16))) & u(0x1F0000FF0000FF)
    x = (x | (x << u(8))) & u(0x100F00F00F00F00F)
    x = (x | (x << u(4))) & u(0x10C30C30C30C30C3)
    x = (x | (x << u(2))) & u(0x1249249249249249)
    return x


def morton_order(points: np.ndarray) -> np.ndarray:
    p = points.astype(np.float64)
    mn = p.min(axis=0)
    ext = max(float((p.max(axis=0) - mn).max()), 1e-12)
    q = ((p - mn) * ((2 ** 21 - 1) / ext)).astype(np.uint64)
    code = (_part1by2(q[:, 0])
            | (_part1by2(q[:, 1]) << np.uint64(1))
            | (_part1by2(q[:, 2]) << np.uint64(2)))
    return np.argsort(code, kind="stable")


def vertex_normals(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = pos.astype(np.float64)
    fn = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        idx = tris[:, k]
        for c in range(3):
            n[:, c] += np.bincount(idx, weights=fn[:, c], minlength=len(p))
    ln = np.linalg.norm(n, axis=1)
    bad = ln < 1e-12
    n[bad] = (0.0, 0.0, 1.0)
    ln[bad] = 1.0
    return (n / ln[:, None]).astype(np.float32)


def simplify_grid(pos: np.ndarray, tris: np.ndarray, uvs: np.ndarray, cell: float):
    """Vertex clustering: snap vertices to a grid, average per cell, drop collapsed
    and duplicate triangles.  O(N), fully vectorised.

    Returns (new_positions, new_uvs, new_tris) or (None, None, None) if collapsed.
    """
    p = pos.astype(np.float64)
    q = np.floor((p - p.min(axis=0)) / cell).astype(np.int64)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv).astype(np.float64)
    newpos = np.stack([np.bincount(inv, weights=p[:, c]) / cnt for c in range(3)], axis=1)

    # Average UVs per cluster cell (same inv mapping)
    nu = uvs.shape[1] if uvs.ndim > 1 else 1
    if nu == 2:
        newuvs = np.stack([np.bincount(inv, weights=uvs[:, c]) / cnt for c in range(2)], axis=1)
    else:
        newuvs = np.bincount(inv, weights=uvs[:, 0]) / cnt[:, None]

    nt = inv[tris]
    ok = (nt[:, 0] != nt[:, 1]) & (nt[:, 1] != nt[:, 2]) & (nt[:, 0] != nt[:, 2])
    nt = nt[ok]
    if len(nt) == 0:
        return None, None, None

    nv = len(newpos)
    s = np.sort(nt, axis=1).astype(np.int64)
    if nv < 2_000_000:
        k = (s[:, 0] * nv + s[:, 1]) * nv + s[:, 2]
        _, first = np.unique(k, return_index=True)
    else:
        _, first = np.unique(s, axis=0, return_index=True)
    nt = nt[np.sort(first)]

    used, inv2 = np.unique(nt.reshape(-1), return_inverse=True)
    return (newpos[used].astype(np.float32),
            newuvs[used].astype(np.float32),
            inv2.reshape(-1, 3).astype(np.int32))


def _make_level(pos, tris, uvs, cluster_size, error) -> LODLevel:
    pos = np.ascontiguousarray(pos, np.float32)
    tris = np.ascontiguousarray(tris, np.int32)
    uvs = np.ascontiguousarray(uvs, np.float32)
    tp = pos[tris]                                   # (T,3,3)
    order = morton_order(tp.mean(axis=1))
    tris = np.ascontiguousarray(tris[order])
    tp = tp[order]
    uvs = uvs[order]
    T = len(tris)

    starts = np.arange(0, T, cluster_size, dtype=np.int64)
    cstart = np.append(starts, T).astype(np.int32)
    counts = np.diff(cstart)

    cmin = np.minimum.reduceat(tp.min(axis=1), starts, axis=0)
    cmax = np.maximum.reduceat(tp.max(axis=1), starts, axis=0)
    centers = (cmin + cmax) * 0.5
    radii = np.linalg.norm(cmax - cmin, axis=1) * 0.5

    fn = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]).astype(np.float64)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    axis = np.add.reduceat(fn, starts, axis=0)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-20)
    dots = np.einsum("ij,ij->i", fn, np.repeat(axis, counts, axis=0))
    mindot = np.minimum.reduceat(dots, starts)
    cutoff = np.where(mindot >= 0.1, np.sqrt(np.clip(1.0 - mindot ** 2, 0.0, 1.0)), 1.0)

    return LODLevel(
        positions=pos, normals=vertex_normals(pos, tris), tris=tris, uvs=uvs,
        cluster_start=cstart, centers=centers.astype(np.float32),
        radii=radii.astype(np.float32), cone_axis=axis.astype(np.float32),
        cone_cutoff=cutoff.astype(np.float32), error=float(error))


def build_asset(positions, tris, cluster_size=128, max_levels=6, key="", uvs=None) -> ClusterAsset:
    pos = np.ascontiguousarray(positions, np.float32).reshape(-1, 3)
    tris = np.ascontiguousarray(tris, np.int32).reshape(-1, 3)
    if len(tris) == 0:
        raise ValueError("mesh has no triangles")
    if uvs is not None:
        uvs = np.ascontiguousarray(uvs, np.float32).reshape(-1, 2)
    else:
        uvs = np.zeros(len(pos), np.float32)

    bmin, bmax = pos.min(axis=0), pos.max(axis=0)
    extent = max(float((bmax - bmin).max()), 1e-9)
    center = (bmin + bmax) * 0.5
    radius = float(np.linalg.norm(bmax - bmin) * 0.5)

    levels = [_make_level(pos, tris, cluster_size, 0.0)]
    prev_T = len(tris)
    res = max(8.0, math.sqrt(prev_T / 2.0))
    for _ in range(1, max_levels):
        res *= 0.5
        if res < 3.0 or prev_T <= cluster_size:
            break
        cell = extent / res
        p2, u2, t2 = simplify_grid(pos, tris, uvs, cell)
        if t2 is None:
            break
        if len(t2) > prev_T * 0.8:      # barely reduced: jump to a coarser grid
            continue
        levels.append(_make_level(p2, t2, u2, cluster_size, cell * 0.866))
        prev_T = len(t2)
    return ClusterAsset(key, levels, center, radius, cluster_size)


# ── disk cache (one .npz per unique geometry) ─────────────────────────────

def save_asset(asset: ClusterAsset, path: str):
    d = {
        "meta": np.array([FORMAT_VERSION, len(asset.levels), asset.cluster_size], np.int64),
        "errors": asset.errors,
        "bounds": np.concatenate([asset.center, [asset.radius]]),
    }
    for i, l in enumerate(asset.levels):
        for f in _FIELDS:
            d[f"l{i}_{f}"] = getattr(l, f)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


def load_asset(path: str, key: str):
    try:
        with np.load(path) as z:
            ver, n, k = (int(v) for v in z["meta"])
            if ver != FORMAT_VERSION:
                return None
            errors = z["errors"]
            b = z["bounds"]
            levels = []
            for i in range(n):
                kw = {f: z[f"l{i}_{f}"] for f in _FIELDS}
                levels.append(LODLevel(error=float(errors[i]), **kw))
        return ClusterAsset(key, levels, b[:3], b[3], k)
    except Exception:
        return None
