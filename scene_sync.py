"""
Scene synchronisation: finds eligible meshes, maps them to shared cluster assets and
builds missing assets asynchronously.

Flow
  import / duplicate object
    -> depsgraph handler flags the scene dirty
    -> timer: content digest of the mesh (cheap: vertex coords + loop indices)
         * asset in memory?  -> instance created immediately (duplicates land here, no rebuild)
         * asset on disk?    -> loaded in a worker thread
         * otherwise         -> triangles extracted on the main thread, built in a worker
    -> object keeps its native display until its asset is ready, then flips to BOUNDS
       display so Blender stops drawing the full mesh and our GPU path takes over.
"""
from __future__ import annotations

import hashlib
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import bpy
import numpy as np

from . import clustering

MARK = "_gpuc_orig_display"


def default_cache_dir() -> str:
    return os.path.join(bpy.utils.user_resource("CONFIG", create=True), "gpu_cluster_cache")


class Settings:
    __slots__ = ("cluster_size", "lod_levels", "min_polys", "workers",
                 "hide_native", "cache_dir")

    @classmethod
    def from_scene(cls, scene):
        p = scene.gpu_cluster
        s = cls()
        s.cluster_size = int(p.cluster_size)
        s.lod_levels = int(p.lod_levels)
        s.min_polys = int(p.min_polys)
        s.workers = int(p.workers)
        s.hide_native = bool(p.hide_native)
        s.cache_dir = bpy.path.abspath(p.cache_dir) if p.cache_dir else default_cache_dir()
        return s

    def build_sig(self):
        return (self.cluster_size, self.lod_levels)


class Instance:
    __slots__ = ("name", "key", "asset", "matrix", "center_w", "radius_w", "scale", "cull")

    def __init__(self, name, key, asset):
        self.name, self.key, self.asset = name, key, asset
        self.cull = None            # renderer-owned partial-draw cache
        self.set_matrix(np.eye(4))

    def set_matrix(self, mw):
        m = np.array(mw, dtype=np.float64)
        self.matrix = m
        self.center_w = m[:3, :3] @ self.asset.center + m[:3, 3]
        self.scale = float(np.linalg.norm(m[:3, :3], axis=0).max())
        self.radius_w = self.asset.radius * self.scale
        self.cull = None


# ── worker jobs (numpy only, never touch bpy) ─────────────────────────────

def _job_build(key, positions, tris, cluster_size, levels, path):
    asset = clustering.build_asset(positions, tris, cluster_size, levels, key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clustering.save_asset(asset, path)
    except OSError:
        pass
    return asset


def _job_load(key, path):
    return clustering.load_asset(path, key)


# ── object helpers ────────────────────────────────────────────────────────

def _hide_native(obj):
    if MARK not in obj:
        obj[MARK] = obj.display_type
    if obj.display_type != "BOUNDS":
        obj.display_type = "BOUNDS"


def release_native(obj):
    if MARK in obj:
        try:
            obj.display_type = obj[MARK]
        except TypeError:
            obj.display_type = "TEXTURED"
        del obj[MARK]


def _digest(mesh) -> str:
    nv, nl, npoly = len(mesh.vertices), len(mesh.loops), len(mesh.polygons)
    co = np.empty(nv * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    vi = np.empty(nl, np.int32)
    mesh.loops.foreach_get("vertex_index", vi)
    h = hashlib.blake2b(digest_size=12)
    h.update(co)
    h.update(vi)
    try:
        lt = np.empty(npoly, np.int32)
        mesh.polygons.foreach_get("loop_total", lt)
        h.update(lt)
    except Exception:
        pass
    h.update(np.array([nv, nl, npoly], np.int64))
    return h.hexdigest()


class SceneSync:
    def __init__(self):
        self.active = False
        self.assets: Dict[str, clustering.ClusterAsset] = {}
        self.instances: Dict[str, Instance] = {}
        self.futures: Dict[str, object] = {}
        self.paths: Dict[str, str] = {}
        self.errors: Dict[str, str] = {}
        self.last_used: Dict[str, float] = {}
        self.version = 0     # instance set changed
        self.xver = 0        # some transform changed
        self.dirty = True
        self._memo = {}      # mesh session_uid -> (digest, nverts, nloops)
        self._pool = None
        self._pool_workers = 0
        self._last_full = 0.0
        self._last_sig = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def activate(self):
        self.active = True
        self.dirty = True
        if not bpy.app.timers.is_registered(self._tick):
            bpy.app.timers.register(self._tick, first_interval=0.05, persistent=True)

    def deactivate(self):
        self.active = False
        if bpy.app.timers.is_registered(self._tick):
            bpy.app.timers.unregister(self._tick)
        self.restore_all()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self.futures.clear()
        self.instances.clear()
        self.version += 1

    def reset_session(self):
        """After file load: uids and object pointers are stale."""
        self._memo.clear()
        self.instances.clear()
        self.version += 1
        self.dirty = True

    def drop_assets(self, purge_disk_dir=None):
        self.assets.clear()
        self.errors.clear()
        self.instances.clear()
        self._memo.clear()
        self.version += 1
        self.dirty = True
        if purge_disk_dir and os.path.isdir(purge_disk_dir):
            for f in os.listdir(purge_disk_dir):
                if f.endswith(".npz"):
                    try:
                        os.remove(os.path.join(purge_disk_dir, f))
                    except OSError:
                        pass

    @staticmethod
    def restore_all():
        for o in bpy.data.objects:
            release_native(o)

    # ── depsgraph hook (must stay cheap) ──────────────────────────────────
    def on_depsgraph(self, depsgraph):
        if not self.active:
            return
        moved = False
        for u in depsgraph.updates:
            idb = u.id
            if isinstance(idb, bpy.types.Object):
                o = idb.original
                inst = self.instances.get(o.name)
                if u.is_updated_transform and inst is not None:
                    inst.set_matrix(o.matrix_world)
                    moved = True
                if u.is_updated_geometry or not u.is_updated_transform:
                    self.dirty = True
            elif isinstance(idb, bpy.types.Mesh):
                if u.is_updated_geometry:
                    self._memo.pop(idb.original.session_uid, None)
                self.dirty = True
            else:
                self.dirty = True
        if moved:
            self.xver += 1

    # ── timer ─────────────────────────────────────────────────────────────
    def _tick(self):
        try:
            scene = bpy.context.scene
            if not self.active or scene is None or not hasattr(scene, "gpu_cluster"):
                return 0.5
            if not scene.gpu_cluster.enabled:
                return 0.5
            st = Settings.from_scene(scene)
            sig = st.build_sig()
            if sig != self._last_sig:
                self._last_sig = sig
                self.dirty = True
            finished = self._poll()
            now = time.time()
            changed = finished
            if self.dirty or finished or now - self._last_full > 1.0:
                self.dirty = False
                self._last_full = now
                changed |= self.sync(scene, st)
            if changed:
                self.tag_redraw()
            return 0.1
        except Exception:
            traceback.print_exc()
            return 0.5

    @staticmethod
    def tag_redraw():
        wm = bpy.context.window_manager
        if wm is None:
            return
        for w in wm.windows:
            for a in w.screen.areas:
                if a.type == "VIEW_3D":
                    a.tag_redraw()

    # ── async jobs ────────────────────────────────────────────────────────
    def _get_pool(self, workers):
        if self._pool is None or workers != self._pool_workers:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
            self._pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                            thread_name_prefix="gpuc")
            self._pool_workers = workers
        return self._pool

    def _poll(self) -> bool:
        done = [k for k, f in self.futures.items() if f.done()]
        for k in done:
            f = self.futures.pop(k)
            try:
                a = f.result()
            except Exception as e:
                self.errors[k] = repr(e)
                continue
            if a is None:                       # corrupt / outdated disk entry
                try:
                    os.remove(self.paths[k])
                except OSError:
                    pass
            else:
                self.assets[k] = a
        return bool(done)

    def _request(self, key, mesh, st: Settings):
        if key in self.futures or key in self.errors:
            return
        path = os.path.join(st.cache_dir, key + ".npz")
        self.paths[key] = path
        pool = self._get_pool(st.workers)
        if os.path.exists(path):
            self.futures[key] = pool.submit(_job_load, key, path)
            return
        mesh.calc_loop_triangles()
        nt = len(mesh.loop_triangles)
        if nt == 0:
            self.errors[key] = "no triangles"
            return
        co = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", co)
        tris = np.empty(nt * 3, np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        self.futures[key] = pool.submit(_job_build, key, co.reshape(-1, 3),
                                        tris.reshape(-1, 3), st.cluster_size,
                                        st.lod_levels, path)

    # ── scene scan ────────────────────────────────────────────────────────
    @staticmethod
    def _eligible(obj, st: Settings) -> bool:
        if obj.type != "MESH" or obj.mode != "OBJECT" or not obj.visible_get():
            return False
        if not (obj.display_type in ("TEXTURED", "SOLID")
                or (MARK in obj and obj.display_type == "BOUNDS")):
            return False
        mesh = obj.data
        if mesh is None or len(mesh.polygons) < st.min_polys or mesh.shape_keys:
            return False
        for m in obj.modifiers:             # evaluated meshes are not clustered
            if m.show_viewport:
                return False
        return True

    def _key_for(self, mesh, st: Settings) -> str:
        uid = mesh.session_uid
        sizes = (len(mesh.vertices), len(mesh.loops))
        m = self._memo.get(uid)
        if m is None or m[1:] != sizes:
            m = (_digest(mesh),) + sizes
            self._memo[uid] = m
        return f"{m[0]}_{st.cluster_size}_{st.lod_levels}"

    def sync(self, scene, st: Settings) -> bool:
        seen = set()
        changed = False
        now = time.time()
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            if not self._eligible(obj, st):
                release_native(obj)
                continue
            key = self._key_for(obj.data, st)
            asset = self.assets.get(key)
            if asset is None:
                self._request(key, obj.data, st)
                release_native(obj)             # stay native until ready
                continue
            self.last_used[key] = now
            inst = self.instances.get(obj.name)
            if inst is None or inst.key != key:
                inst = Instance(obj.name, key, asset)
                inst.set_matrix(obj.matrix_world)
                self.instances[obj.name] = inst
                changed = True
            seen.add(obj.name)
            if st.hide_native:
                _hide_native(obj)
            else:
                release_native(obj)

        gone = [n for n in self.instances if n not in seen]
        for n in gone:
            del self.instances[n]
        if gone or changed:
            self.version += 1
            changed = True

        # evict assets nobody has used for a while
        pending = set(self.futures)
        for k in list(self.assets):
            if k not in pending and now - self.last_used.get(k, now) > 120.0:
                del self.assets[k]
                self.last_used.pop(k, None)
        return changed

    def status(self) -> dict:
        tris = sum(i.asset.tri_count for i in self.instances.values())
        return {
            "instances": len(self.instances),
            "assets": len(self.assets),
            "pending": len(self.futures),
            "errors": len(self.errors),
            "source_tris": tris,
        }
