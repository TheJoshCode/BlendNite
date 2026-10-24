"""
BlendNite core: pure numpy (no bpy), safe to run in worker threads.

Per asset (one unique mesh geometry, stored in object space so duplicates share it):

  * LOD 0 is the original mesh.  Corners are welded on (vertex, uv, flat-face) so UV
    seams and flat-shaded faces render correctly.  Coarser LODs come from grid vertex
    clustering on a FIXED grid origin at halving resolution (~4x fewer triangles each).
  * Every LOD is split into spatially coherent clusters.  Triangles are sorted by
    (material slot, Morton code) and chunked so that
        - a cluster never mixes material slots     -> materials are per cluster
        - a cluster never exceeds `max_verts` unique vertices (guaranteed by splitting)
  * Each cluster stores a bounding sphere and a normal cone.

Incremental edits (`patch_asset`):
  The asset remembers the source vertex positions.  When only positions changed
  (topology / UVs / materials identical) we diff against the old positions and
    - LOD 0: update moved vertices, recompute normals of the affected neighbourhood,
      recompute bounds + cones ONLY for clusters containing a moved vertex.
    - coarse LODs: if every moved vertex stays in its grid cell, re-average only the
      touched cells and again patch only the touched clusters; otherwise just that one
      level is rebuilt.
  Untouched arrays are shared with the old asset (copy-on-write), so duplicates that
  still use the old asset are never affected.
"""
from __future__ import annotations

import math
import os

import numpy as np

FORMAT_VERSION = 4
CELL_OFF = 1 << 20            # grid cell coordinates are offset so edits may go negative

_LEVEL_FIELDS = (
    "positions", "normals", "uvs", "tris", "tri_mat", "tri_smooth",
    "cluster_start", "cluster_mat", "centers", "radii", "cone_axis", "cone_cutoff",
    "src",          # LOD0 only: source-mesh vertex of every welded vertex
    "vmap",         # coarse only: LOD0 vertex -> this level's vertex (-1 = dropped)
    "cell_keys",    # coarse only: grid cell key of every vertex of this level
)


class LODLevel:
    __slots__ = _LEVEL_FIELDS + ("error", "cell")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def replace(self, **kw) -> "LODLevel":
        d = {k: getattr(self, k) for k in self.__slots__}
        d.update(kw)
        return LODLevel(**d)

    @property
    def n_tris(self) -> int:
        return int(len(self.tris))

    @property
    def n_verts(self) -> int:
        return int(len(self.positions))

    @property
    def n_clusters(self) -> int:
        return int(len(self.cluster_start) - 1)

    def material_groups(self):
        """[(material_slot, first_cluster, end_cluster)] - clusters are sorted by slot."""
        cm = self.cluster_mat
        if len(cm) == 0:
            return []
        b = np.flatnonzero(np.diff(cm)) + 1
        starts = np.concatenate([[0], b]).astype(np.int64)
        ends = np.concatenate([b, [len(cm)]]).astype(np.int64)
        return [(int(cm[s]), int(s), int(e)) for s, e in zip(starts, ends)]


class ClusterAsset:
    def __init__(self, key, levels, center, radius, max_verts, src_co, origin, topo):
        self.key = key
        self.levels = levels
        self.center = np.asarray(center, np.float64)
        self.radius = float(radius)
        self.max_verts = int(max_verts)
        self.src_co = src_co            # (Ns,3) float32 source positions (for edit diffing)
        self.origin = np.asarray(origin, np.float64)
        self.topo = topo

    @property
    def errors(self) -> np.ndarray:
        return np.array([l.error for l in self.levels], np.float64)

    @property
    def vert_counts(self) -> np.ndarray:
        return np.array([l.n_verts for l in self.levels], np.int64)

    @property
    def tri_count(self) -> int:
        return self.levels[0].n_tris


# ── small helpers ─────────────────────────────────────────────────────────

def _part1by2(x):
    u = np.uint64
    x = x & u(0x1FFFFF)
    x = (x | (x << u(32))) & u(0x1F00000000FFFF)
    x = (x | (x << u(16))) & u(0x1F0000FF0000FF)
    x = (x | (x << u(8))) & u(0x100F00F00F00F00F)
    x = (x | (x << u(4))) & u(0x10C30C30C30C30C3)
    x = (x | (x << u(2))) & u(0x1249249249249249)
    return x


def morton_codes(points: np.ndarray) -> np.ndarray:
    p = points.astype(np.float64)
    mn = p.min(axis=0)
    ext = max(float((p.max(axis=0) - mn).max()), 1e-12)
    q = ((p - mn) * ((2 ** 21 - 1) / ext)).astype(np.uint64)
    return (_part1by2(q[:, 0])
            | (_part1by2(q[:, 1]) << np.uint64(1))
            | (_part1by2(q[:, 2]) << np.uint64(2)))


def _normalize(n: np.ndarray) -> np.ndarray:
    n = n.astype(np.float64)
    ln = np.linalg.norm(n, axis=1)
    bad = ln < 1e-12
    n[bad] = (0.0, 0.0, 1.0)
    ln[bad] = 1.0
    return (n / ln[:, None]).astype(np.float32)


def _cell_keys(p: np.ndarray, origin: np.ndarray, cell: float) -> np.ndarray:
    q = np.floor((p.astype(np.float64) - origin) / cell).astype(np.int64) + CELL_OFF
    np.clip(q, 0, (1 << 21) - 1, out=q)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def update_normals(normals, pos, tris, smooth, weld, tri_aff):
    """Recompute, in place, the normals of every weld group touched by a triangle in
    `tri_aff`.  Smooth faces are area-weighted per weld group (so UV seams stay smooth);
    flat faces give their face normal to their own (unshared) vertices."""
    W = weld[tris]
    nW = int(weld.max()) + 1 if len(weld) else 0
    wmask = np.zeros(nW, bool)
    wmask[W[tri_aff].ravel()] = True
    ta = np.flatnonzero(wmask[W].any(axis=1))
    if len(ta) == 0:
        return
    p = pos.astype(np.float64)
    t = tris[ta]
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    sm = smooth[ta]
    fs = fn * sm[:, None]
    acc = np.zeros((nW, 3))
    for k in range(3):
        for c in range(3):
            acc[:, c] += np.bincount(W[ta, k], weights=fs[:, c], minlength=nW)
    vsel = np.flatnonzero(wmask[weld])
    normals[vsel] = _normalize(acc[weld[vsel]])
    ft = np.flatnonzero(~sm)
    if len(ft):
        fnn = _normalize(fn[ft])
        for k in range(3):
            normals[t[ft, k]] = fnn


def _cluster_stats(pos, tris, cs, cids):
    """Bounding sphere + normal cone for the clusters `cids` only."""
    if len(cids) == 0:
        z3 = np.zeros((0, 3), np.float32)
        return z3, np.zeros(0, np.float32), z3, np.zeros(0, np.float32)
    cs = cs.astype(np.int64)
    counts = cs[cids + 1] - cs[cids]
    offs = np.cumsum(counts) - counts
    idx = (np.arange(int(counts.sum()), dtype=np.int64)
           - np.repeat(offs, counts) + np.repeat(cs[cids], counts))
    tp = pos[tris[idx]]                                         # (n,3,3) float32
    cmin = np.minimum.reduceat(tp.min(axis=1), offs, axis=0)
    cmax = np.maximum.reduceat(tp.max(axis=1), offs, axis=0)
    centers = (cmin + cmax) * 0.5
    radii = np.linalg.norm(cmax - cmin, axis=1) * 0.5
    t64 = tp.astype(np.float64)
    fn = np.cross(t64[:, 1] - t64[:, 0], t64[:, 2] - t64[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    axis = np.add.reduceat(fn, offs, axis=0)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-20)
    dots = np.einsum("ij,ij->i", fn, np.repeat(axis, counts, axis=0))
    mindot = np.minimum.reduceat(dots, offs)
    cutoff = np.where(mindot >= 0.1, np.sqrt(np.clip(1.0 - mindot ** 2, 0.0, 1.0)), 1.0)
    return (centers.astype(np.float32), radii.astype(np.float32),
            axis.astype(np.float32), cutoff.astype(np.float32))


def _cluster_bounds(tris, tri_mat, max_verts) -> np.ndarray:
    """Triangle offsets of cluster boundaries.  Never crosses a material change and
    guarantees <= max_verts unique vertices per cluster by recursive halving."""
    T = len(tris)
    t0 = max(4, int(max_verts * 1.4))
    mb = np.flatnonzero(np.diff(tri_mat)) + 1
    gs = np.concatenate([[0], mb, [T]]).astype(np.int64)
    parts = [np.arange(a, b, t0, dtype=np.int64) for a, b in zip(gs[:-1], gs[1:])]
    bnd = np.unique(np.concatenate(parts + [np.array([T], np.int64)]))
    nV = int(tris.max()) + 1
    t64 = tris.astype(np.int64)
    while True:
        counts = np.diff(bnd)
        C = len(counts)
        cid = np.repeat(np.arange(C, dtype=np.int64), counts)
        u = np.unique((cid[:, None] * nV + t64).ravel())
        vc = np.bincount(u // nV, minlength=C)
        bad = (vc > max_verts) & (counts > 1)
        if not bad.any():
            break
        bnd = np.union1d(bnd, bnd[:-1][bad] + counts[bad] // 2)
    return bnd.astype(np.int32)


def _make_level(pos, uvs, tris, tri_mat, tri_smooth, weld, src, vmap, cell_keys,
                max_verts, error, cell) -> LODLevel:
    pos = np.ascontiguousarray(pos, np.float32)
    uvs = np.ascontiguousarray(uvs, np.float32)
    tris = np.ascontiguousarray(tris, np.int32)
    tri_mat = np.asarray(tri_mat, np.int32)
    tri_smooth = np.asarray(tri_smooth, bool)
    code = morton_codes(pos[tris].mean(axis=1))
    order = np.lexsort((code, tri_mat))                 # material first, Morton within
    tris = np.ascontiguousarray(tris[order])
    tri_mat = np.ascontiguousarray(tri_mat[order])
    tri_smooth = np.ascontiguousarray(tri_smooth[order])

    cs = _cluster_bounds(tris, tri_mat, max_verts)
    cids = np.arange(len(cs) - 1)
    centers, radii, axis, cutoff = _cluster_stats(pos, tris, cs, cids)

    normals = np.zeros((len(pos), 3), np.float32)
    update_normals(normals, pos, tris, tri_smooth, weld, np.ones(len(tris), bool))

    return LODLevel(
        positions=pos, normals=normals, uvs=uvs, tris=tris, tri_mat=tri_mat,
        tri_smooth=tri_smooth, cluster_start=cs, cluster_mat=tri_mat[cs[:-1]].copy(),
        centers=centers, radii=radii, cone_axis=axis, cone_cutoff=cutoff,
        src=src, vmap=vmap, cell_keys=cell_keys, error=float(error), cell=float(cell))


# ── corner welding ────────────────────────────────────────────────────────

def weld_corners(co, tv, tl, ts, uv):
    """Unique render vertices = unique (source vertex, uv, flat-face id).
    Returns (src, uv0, tris0)."""
    nt = len(tv)
    flat_any = not bool(ts.all())
    if uv is None and not flat_any:
        return (np.arange(len(co), dtype=np.int32),
                np.zeros((len(co), 2), np.float32), tv.astype(np.int32))
    rows = np.zeros((nt * 3, 4), np.int32)
    rows[:, 0] = tv.reshape(-1)
    if uv is not None:
        rows[:, 1:3] = np.ascontiguousarray(uv[tl.reshape(-1)], np.float32).view(np.int32)
    rows[:, 3] = -1
    if flat_any:
        ids = np.repeat(np.arange(nt, dtype=np.int32), 3)
        rows[:, 3] = np.where(np.repeat(~ts, 3), ids, -1)
    v = rows.view(np.dtype((np.void, 16))).reshape(-1)
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    src = tv.reshape(-1)[first].astype(np.int32)
    uv0 = (uv[tl.reshape(-1)[first]] if uv is not None
           else np.zeros((len(first), 2), np.float32))
    return src, uv0.astype(np.float32), inv.reshape(-1, 3).astype(np.int32)


def simplify_grid(pos, uvs, tris, origin, cell):
    """Vertex clustering on a fixed grid.  Returns
    (positions, uvs, tris, tri_index, vmap, cell_keys) or None if everything collapsed."""
    p = pos.astype(np.float64)
    key = _cell_keys(p, origin, cell)
    ukeys, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    nc = len(ukeys)
    cnt = np.bincount(inv, minlength=nc).astype(np.float64)
    newpos = np.stack([np.bincount(inv, weights=p[:, c], minlength=nc) / cnt
                       for c in range(3)], axis=1)
    newuv = np.stack([np.bincount(inv, weights=uvs[:, c].astype(np.float64), minlength=nc) / cnt
                      for c in range(2)], axis=1)

    nt = inv[tris]
    ok = (nt[:, 0] != nt[:, 1]) & (nt[:, 1] != nt[:, 2]) & (nt[:, 0] != nt[:, 2])
    tri_idx = np.flatnonzero(ok)
    nt = nt[ok]
    if len(nt) == 0:
        return None
    s = np.sort(nt, axis=1).astype(np.int64)
    if nc < 2_000_000:
        _, first = np.unique((s[:, 0] * nc + s[:, 1]) * nc + s[:, 2], return_index=True)
    else:
        _, first = np.unique(s, axis=0, return_index=True)
    first = np.sort(first)
    nt, tri_idx = nt[first], tri_idx[first]

    used = np.unique(nt.reshape(-1))
    remap = np.full(nc, -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return (newpos[used].astype(np.float32), newuv[used].astype(np.float32),
            remap[nt].astype(np.int32), tri_idx, remap[inv], ukeys[used])


def _coarse_level(p2, u2, t2, tri_mat0, ti, vmap, ck, max_verts, cell):
    return _make_level(p2, u2, t2, tri_mat0[ti], np.ones(len(t2), bool),
                       weld=np.arange(len(p2), dtype=np.int32),
                       src=np.zeros(0, np.int32), vmap=vmap, cell_keys=ck,
                       max_verts=max_verts, error=cell * 0.866, cell=cell)


# ── build ─────────────────────────────────────────────────────────────────

def build_asset(co, tv, tl, tm, ts, uv, max_verts=128, max_levels=8, key="", topo=""):
    """co (Ns,3) source positions; tv/tl (T,3) per-triangle source vertex / loop
    indices; tm (T,) material slot; ts (T,) smooth flag; uv (loops,2) or None."""
    co = np.ascontiguousarray(co, np.float32).reshape(-1, 3)
    tv = np.ascontiguousarray(tv, np.int32).reshape(-1, 3)
    tl = np.ascontiguousarray(tl, np.int32).reshape(-1, 3)
    tm = np.ascontiguousarray(tm, np.int32).reshape(-1)
    ts = np.ascontiguousarray(ts, bool).reshape(-1)
    if len(tv) == 0:
        raise ValueError("mesh has no triangles")
    if uv is not None:
        uv = np.ascontiguousarray(uv, np.float32).reshape(-1, 2)
    max_verts = int(max(12, max_verts))

    src, uv0, tris0 = weld_corners(co, tv, tl, ts, uv)
    pos0 = co[src]
    bmin, bmax = pos0.min(axis=0), pos0.max(axis=0)
    extent = max(float((bmax - bmin).max()), 1e-9)
    center = (bmin + bmax) * 0.5
    radius = float(np.linalg.norm(bmax - bmin) * 0.5)
    origin = bmin.astype(np.float64)

    lvl0 = _make_level(pos0, uv0, tris0, tm, ts, weld=src, src=src,
                       vmap=np.zeros(0, np.int32), cell_keys=np.zeros(0, np.int64),
                       max_verts=max_verts, error=0.0, cell=0.0)
    levels = [lvl0]
    prev_T = lvl0.n_tris
    res = max(8.0, math.sqrt(prev_T / 2.0))
    for _ in range(1, max_levels):
        res *= 0.5
        if res < 3.0 or prev_T <= max_verts:
            break
        cell = extent / res
        r = simplify_grid(pos0, uv0, lvl0.tris, origin, cell)
        if r is None:
            break
        p2, u2, t2, ti, vmap, ck = r
        if len(t2) > prev_T * 0.8:          # barely reduced: jump to a coarser grid
            continue
        levels.append(_coarse_level(p2, u2, t2, lvl0.tri_mat, ti, vmap, ck, max_verts, cell))
        prev_T = len(t2)
    return ClusterAsset(key, levels, center, radius, max_verts, co.copy(), origin, topo)


# ── incremental edit patching ─────────────────────────────────────────────

def _patch_level(L: LODLevel, newpos, vmask, weld) -> LODLevel:
    """Copy-on-write update of one level after the vertices flagged in `vmask` moved."""
    tri_aff = vmask[L.tris].any(axis=1)
    if not tri_aff.any():
        return L.replace(positions=newpos)
    normals = L.normals.copy()
    update_normals(normals, newpos, L.tris, L.tri_smooth, weld, tri_aff)
    cs = L.cluster_start
    tri_cluster = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
    cids = np.unique(tri_cluster[tri_aff])
    c, r, a, k = _cluster_stats(newpos, L.tris, cs, cids)
    centers, radii = L.centers.copy(), L.radii.copy()
    axis, cutoff = L.cone_axis.copy(), L.cone_cutoff.copy()
    centers[cids], radii[cids], axis[cids], cutoff[cids] = c, r, a, k
    return L.replace(positions=newpos, normals=normals, centers=centers, radii=radii,
                     cone_axis=axis, cone_cutoff=cutoff)


def _patch_coarse(L, L0, pos0, chg0, origin, max_verts):
    vm = L.vmap[chg0]
    rebuild = bool((vm < 0).any())
    if not rebuild:
        nk = _cell_keys(pos0[chg0], origin, L.cell)
        rebuild = bool((nk != L.cell_keys[vm]).any())      # a vertex changed grid cell
    if rebuild:                                            # topology of this level changes
        r = simplify_grid(pos0, L0.uvs, L0.tris, origin, L.cell)
        if r is None:
            return None
        p2, u2, t2, ti, vmap, ck = r
        return _coarse_level(p2, u2, t2, L0.tri_mat, ti, vmap, ck, max_verts, L.cell)

    V = L.n_verts
    aff = np.unique(vm)
    amask = np.zeros(V, bool)
    amask[aff] = True
    vmap = L.vmap
    members = np.flatnonzero((vmap >= 0) & amask[np.maximum(vmap, 0)])
    mv = vmap[members]
    cnt = np.bincount(mv, minlength=V).astype(np.float64)
    newp = L.positions.copy()
    for c in range(3):
        s = np.bincount(mv, weights=pos0[members, c].astype(np.float64), minlength=V)
        newp[aff, c] = (s[aff] / cnt[aff]).astype(np.float32)
    return _patch_level(L, newp, amask, np.arange(V, dtype=np.int32))


def patch_asset(old: ClusterAsset, co_new, new_key, new_topo, max_changed_frac=0.4):
    """New asset for `old` after vertex positions changed (same topology).
    Returns None when a full rebuild is the better choice."""
    co_new = np.ascontiguousarray(co_new, np.float32).reshape(-1, 3)
    if co_new.shape != old.src_co.shape:
        return None
    diff = np.any(co_new != old.src_co, axis=1)
    nchg = int(diff.sum())
    L0 = old.levels[0]
    chg0 = np.flatnonzero(diff[L0.src])
    if nchg == 0 or len(chg0) == 0:
        return ClusterAsset(new_key, old.levels, old.center, old.radius, old.max_verts,
                            co_new.copy(), old.origin, new_topo)
    if nchg > max_changed_frac * len(diff):
        return None

    pos0 = L0.positions.copy()
    pos0[chg0] = co_new[L0.src[chg0]]
    vmask = np.zeros(len(pos0), bool)
    vmask[chg0] = True
    levels = [_patch_level(L0, pos0, vmask, L0.src)]
    for L in old.levels[1:]:
        nl = _patch_coarse(L, L0, pos0, chg0, old.origin, old.max_verts)
        if nl is None:
            return None
        levels.append(nl)
    bmin, bmax = pos0.min(axis=0), pos0.max(axis=0)
    return ClusterAsset(new_key, levels, (bmin + bmax) * 0.5,
                        float(np.linalg.norm(bmax - bmin) * 0.5), old.max_verts,
                        co_new.copy(), old.origin, new_topo)


# ── disk cache (one .npz per unique geometry) ─────────────────────────────

_ARRAY_FIELDS = _LEVEL_FIELDS


def save_asset(asset: ClusterAsset, path: str):
    d = {
        "meta": np.array([FORMAT_VERSION, len(asset.levels), asset.max_verts], np.int64),
        "errors": asset.errors,
        "cells": np.array([l.cell for l in asset.levels], np.float64),
        "bounds": np.concatenate([asset.center, [asset.radius]]),
        "origin": asset.origin,
        "src_co": asset.src_co,
        "topo": np.array(asset.topo),
    }
    for i, l in enumerate(asset.levels):
        for f in _ARRAY_FIELDS:
            d[f"l{i}_{f}"] = getattr(l, f)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


def load_asset(path: str, key: str):
    try:
        with np.load(path) as z:
            ver, n, mv = (int(v) for v in z["meta"])
            if ver != FORMAT_VERSION:
                return None
            errors, cells, b = z["errors"], z["cells"], z["bounds"]
            levels = []
            for i in range(n):
                kw = {f: z[f"l{i}_{f}"] for f in _ARRAY_FIELDS}
                levels.append(LODLevel(error=float(errors[i]), cell=float(cells[i]), **kw))
            asset = ClusterAsset(key, levels, b[:3], b[3], mv, z["src_co"],
                                 z["origin"], str(z["topo"]))
        try:
            os.utime(path)              # mark as recently used for cache pruning
        except OSError:
            pass
        return asset
    except Exception:
        return None


def prune_cache(folder: str, max_age_days: float = 30.0):
    import time
    if not os.path.isdir(folder):
        return
    limit = time.time() - max_age_days * 86400.0
    for f in os.listdir(folder):
        if f.endswith(".npz"):
            p = os.path.join(folder, f)
            try:
                if os.path.getmtime(p) < limit:
                    os.remove(p)
            except OSError:
                pass
