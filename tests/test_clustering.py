"""Run: python tests/test_clustering.py   (needs only numpy)"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clustering as C  # noqa: E402


def uv_sphere(n_lat, n_lon):
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    pos = np.stack([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)], -1).reshape(-1, 3)
    idx = np.arange(n_lat * n_lon).reshape(n_lat, n_lon)
    a, b = idx[:-1], idx[1:]
    a2, b2 = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
    t = np.concatenate([np.stack([a, b, a2], -1).reshape(-1, 3),
                        np.stack([a2, b, b2], -1).reshape(-1, 3)])
    return pos.astype(np.float32), t.astype(np.int32)


def main():
    pos, tris = uv_sphere(400, 800)          # ~640k tris
    t0 = time.time()
    a = C.build_asset(pos, tris, 128, 6, "k")
    print(f"built {len(tris):,} tris in {time.time() - t0:.2f}s, levels:",
          [(l.n_tris, l.n_clusters) for l in a.levels])

    tc = [l.n_tris for l in a.levels]
    assert len(tc) >= 3 and all(x > y for x, y in zip(tc, tc[1:])), tc
    assert all(a.levels[i].error < a.levels[i + 1].error for i in range(len(tc) - 1))

    for l in a.levels:
        assert l.tris.min() >= 0 and l.tris.max() < len(l.positions)
        assert l.cluster_start[0] == 0 and l.cluster_start[-1] == l.n_tris
        assert np.isfinite(l.normals).all() and np.isfinite(l.centers).all()
        # every triangle lies inside its cluster sphere
        cid = np.repeat(np.arange(l.n_clusters), np.diff(l.cluster_start))
        d = np.linalg.norm(l.positions[l.tris] - l.centers[cid][:, None], axis=2).max(1)
        assert (d <= l.radii[cid] + 1e-4).all()

    # normal cone culling must be conservative: a culled cluster has no front-facing tri
    rng = np.random.default_rng(1)
    l = a.levels[1]
    tp = l.positions[l.tris]
    fn = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0])
    cid = np.repeat(np.arange(l.n_clusters), np.diff(l.cluster_start))
    culled_total = 0
    for _ in range(20):
        cam = rng.normal(size=3)
        cam = cam / np.linalg.norm(cam) * rng.uniform(1.5, 8)
        v = l.centers - cam
        back = (v * l.cone_axis).sum(1) >= l.cone_cutoff * np.linalg.norm(v, axis=1) + l.radii
        culled_total += back.sum()
        facing = (fn * (tp[:, 0] - cam)).sum(1) < 0      # front-facing triangles
        assert not (facing & back[cid]).any(), "cone culled a visible triangle"
    assert culled_total > 0, "cone culling never triggered"
    print("cone culling conservative, culled clusters:", culled_total)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.npz")
        C.save_asset(a, p)
        b = C.load_asset(p, "k")
        assert b is not None and len(b.levels) == len(a.levels)
        assert np.array_equal(b.levels[2].tris, a.levels[2].tris)
        assert abs(b.radius - a.radius) < 1e-9
    print("npz round trip ok")


if __name__ == "__main__":
    main()
