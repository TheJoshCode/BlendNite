"""
Scene synchronisation: finds eligible meshes, maps them to shared cluster assets and
builds / patches missing assets on worker threads.

  duplicate (Alt+D / Shift+D / re-import)
      -> mesh content digest is identical -> asset already in memory -> instance only
         carries a matrix, nothing is rebuilt.
  new geometry
      -> arrays read on the main thread, clusters + LODs built in a thread pool,
         cached on disk.  The object keeps native drawing until the asset is ready.
  vertex edit (edit mode -> object mode)
      -> topology digest unchanged, positions changed -> only the clusters around the
         moved vertices are recomputed (clustering.patch_asset).  The previous asset keeps
         being drawn until the patch lands, so there is no flash of native geometry.
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

MARK = "_blendnite_orig_display"
MAX_LEVELS = 8
IDLE_EVICT_S = 120.0
WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))
DEFAULT_MATERIAL = ((0.8, 0.8, 0.8, 1.0), None)


def default_cache_dir() -> str:
    """Per-user cache folder.  Extensions (4.2+) must use extension_path_user."""
    try:
        base = bpy.utils.extension_path_user(__package__, path="cache", create=True)
        if base:
            return base
    except Exception:
        pass                                   # legacy add-on install: not an extension
    return os.path.join(bpy.utils.user_resource("CONFIG", create=True), "blendnite_cache")


class Settings:
    __slots__ = ("precision_px", "max_verts", "max_drawn", "min_verts", "cache_dir")

    @classmethod
    def from_scene(cls, scene):
        p = scene.blendnite
        s = cls()
        s.precision_px = float(p.precision_px)
        s.max_verts = int(p.max_cluster_verts)
        s.max_drawn = int(p.max_drawn_verts)
        s.min_verts = int(p.min_verts)
        s.cache_dir = default_cache_dir()
        return s

    def build_sig(self):
        return (self.max_verts,)


# ── materials (main thread) ───────────────────────────────────────────────

def _find_image(sock, depth=0):
    if depth > 4 or not sock.is_linked:
        return None
    node = sock.links[0].from_node
    if node.type == "TEX_IMAGE":
        return node.image
    for s in node.inputs:
        if s.is_linked and s.type == "RGBA":
            img = _find_image(s, depth + 1)
            if img is not None:
                return img
    return None


def _color_socket(nt):
    out = next((n for n in nt.nodes
                if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if out is not None and out.inputs["Surface"].is_linked:
        n = out.inputs["Surface"].links[0].from_node
        for name in ("Base Color", "Color"):
            if name in n.inputs:
                return n.inputs[name]
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n.inputs["Base Color"]
    return None


def material_info(mat):
    """(rgba, image_name | None) approximating the material for the viewport."""
    if mat is None:
        return DEFAULT_MATERIAL
    color = tuple(float(c) for c in mat.diffuse_color)
    image = None
    try:
        if mat.node_tree:                       # Material.use_nodes is deprecated (5.0) / always True
            sock = _color_socket(mat.node_tree)
            if sock is not None:
                if sock.is_linked:
                    img = _find_image(sock)
                    if img is not None:
                        image = img.name
                else:
                    dv = sock.default_value
                    color = (float(dv[0]), float(dv[1]), float(dv[2]), 1.0)
    except Exception:
        pass
    return (color, image)


def _slot_sig(obj):
    return tuple(s.material.name_full if s.material else None for s in obj.material_slots)


# ── instances ─────────────────────────────────────────────────────────────

class Instance:
    __slots__ = ("name", "key", "asset", "matrix", "center_w", "radius_w", "scale",
                 "flip", "cull", "mats", "mat_sig")

    def __init__(self, name, key, asset):
        self.name, self.key, self.asset = name, key, asset
        self.cull = None
        self.mats = []
        self.mat_sig = None
        self.set_matrix(np.eye(4))

    def set_matrix(self, mw):
        m = np.array(mw, dtype=np.float64)
        self.matrix = m
        self.center_w = m[:3, :3] @ self.asset.center + m[:3, 3]
        self.scale = float(np.linalg.norm(m[:3, :3], axis=0).max())
        self.radius_w = self.asset.radius * self.scale
        self.flip = bool(np.linalg.det(m[:3, :3]) < 0)
        self.cull = None


# ── mesh reading (main thread) ────────────────────────────────────────────

def _get(coll, attr, dtype, count):
    a = np.empty(count, dtype)
    coll.foreach_get(attr, a)
    return a


def _uv_array(mesh, nl):
    """Active UV map as (nl, 2) float32, or None.

    MeshUVLoop / MeshUVLoopLayer.data are deprecated (3.5) - use the `uv` attribute
    collection (Float2AttributeValue.vector); `.data` is only a last resort."""
    layer = mesh.uv_layers.active
    if layer is None:
        return None
    try:
        a = np.empty(nl * 2, np.float32)
        layer.uv.foreach_get("vector", a)
        return a.reshape(-1, 2)
    except Exception:
        a = np.empty(nl * 2, np.float32)
        layer.data.foreach_get("uv", a)
        return a.reshape(-1, 2)


def _poly_offsets(mesh, npoly):
    """Face-corner start offsets (npoly + 1), or None if neither API is available."""
    try:
        a = np.empty(npoly + 1, np.int32)
        mesh.polygon_offset_indices.foreach_get(a)
        return a
    except Exception:
        pass
    try:
        return _get(mesh.polygons, "loop_total", np.int32, npoly)
    except Exception:
        return None


def _face_attr(mesh, name, dtype, count, default):
    """Per-face generic attribute (sharp_face, material_index) with a default."""
    at = mesh.attributes.get(name)
    if at is None or at.domain != "FACE" or len(at.data) != count:
        return np.full(count, default, dtype)
    a = np.empty(count, dtype)
    at.data.foreach_get("value", a)
    return a


def mesh_digests(mesh):
    """(topology digest, position digest).  Topology covers connectivity, UVs, face
    smoothing and material assignment: if it is unchanged an edit is position-only."""
    nv, nl, npoly = len(mesh.vertices), len(mesh.loops), len(mesh.polygons)
    ht = hashlib.blake2b(digest_size=12)
    ht.update(np.array([nv, nl, npoly], np.int64))
    ht.update(_get(mesh.loops, "vertex_index", np.int32, nl))
    offs = _poly_offsets(mesh, npoly)
    if offs is not None:
        ht.update(offs)
    ht.update(_face_attr(mesh, "material_index", np.int32, npoly, 0))
    ht.update(_face_attr(mesh, "sharp_face", np.bool_, npoly, False))
    uv = _uv_array(mesh, nl)
    if uv is not None:
        ht.update(uv)
    hc = hashlib.blake2b(digest_size=12)
    hc.update(_get(mesh.vertices, "co", np.float32, nv * 3))
    return ht.hexdigest(), hc.hexdigest()


def extract_mesh(mesh):
    mesh.calc_loop_triangles()
    nt = len(mesh.loop_triangles)
    if nt == 0:
        return None
    nv, nl, npoly = len(mesh.vertices), len(mesh.loops), len(mesh.polygons)
    lt = mesh.loop_triangles
    pi = _get(lt, "polygon_index", np.int32, nt)
    sharp = _face_attr(mesh, "sharp_face", np.bool_, npoly, False)
    mat = _face_attr(mesh, "material_index", np.int32, npoly, 0)
    return {
        "co": _get(mesh.vertices, "co", np.float32, nv * 3).reshape(-1, 3),
        "tv": _get(lt, "vertices", np.int32, nt * 3).reshape(-1, 3),
        "tl": _get(lt, "loops", np.int32, nt * 3).reshape(-1, 3),
        "tm": mat[pi],
        "ts": ~sharp[pi],
        "uv": _uv_array(mesh, nl),
    }


# ── worker jobs (numpy only, never touch bpy) ─────────────────────────────

def _job_build(key, d, max_verts, path, topo):
    asset = clustering.build_asset(d["co"], d["tv"], d["tl"], d["tm"], d["ts"], d["uv"],
                                   max_verts, MAX_LEVELS, key, topo)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clustering.save_asset(asset, path)
    except OSError:
        pass
    return asset


def _job_load(key, path):
    return clustering.load_asset(path, key)


def _job_patch(old, co, key, topo):
    return clustering.patch_asset(old, co, key, topo)


def _job_prune(folder):
    clustering.prune_cache(folder)


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


class SceneSync:
    def __init__(self):
        self.active = False
        self.suspended = False       # some 3D view is not in Solid / Material Preview
        self.assets: Dict[str, clustering.ClusterAsset] = {}
        self.instances: Dict[str, Instance] = {}
        self.obj_key: Dict[str, str] = {}      # object name -> key of its last good asset
        self.futures: Dict[str, object] = {}
        self.kinds: Dict[str, str] = {}        # key -> "load" | "build" | "patch"
        self.fb_key: Dict[str, str] = {}       # patch key -> old key still drawn meanwhile
        self.no_patch = set()
        self.paths: Dict[str, str] = {}
        self.errors: Dict[str, str] = {}
        self.last_used: Dict[str, float] = {}
        self.version = 0     # instance set changed
        self.xver = 0        # some transform changed
        self.mat_epoch = 0   # material / image data changed
        self.dirty = True
        self._mat_dirty = False
        self._memo = {}      # mesh session_uid -> (topo, co, nverts, nloops)
        self._pool = None
        self._pool_workers = 0
        self._last_full = 0.0
        self._last_sig = None
        self._last_mode = None
        self._pruned = False
        self._tick_cb = self._tick           # one stable bound-method object for bpy.app.timers

    # ── lifecycle ─────────────────────────────────────────────────────────
    def activate(self):
        self.active = True
        self.dirty = True
        self._pruned = False
        if not bpy.app.timers.is_registered(self._tick_cb):
            bpy.app.timers.register(self._tick_cb, first_interval=0.05, persistent=True)

    def deactivate(self):
        self.active = False
        if bpy.app.timers.is_registered(self._tick_cb):
            bpy.app.timers.unregister(self._tick_cb)
        self.restore_all()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        for d in (self.futures, self.kinds, self.fb_key, self.errors, self.instances):
            d.clear()
        self.no_patch.clear()
        self.version += 1

    def reset_session(self):
        """After file load: uids and object pointers are stale."""
        self._memo.clear()
        self.instances.clear()
        self.obj_key.clear()
        self.version += 1
        self.dirty = True

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
                if u.is_updated_geometry and o.data is not None:
                    self._memo.pop(o.data.session_uid, None)
                if u.is_updated_geometry or not u.is_updated_transform:
                    self.dirty = True
            elif isinstance(idb, bpy.types.Mesh):
                if u.is_updated_geometry:
                    self._memo.pop(idb.original.session_uid, None)
                self.dirty = True
            elif isinstance(idb, (bpy.types.Material, bpy.types.Image)):
                self._mat_dirty = True
                self.dirty = True
            else:
                self.dirty = True
        if moved:
            self.xver += 1

    # ── timer ─────────────────────────────────────────────────────────────
    @staticmethod
    def _views_ok() -> bool:
        wm = bpy.context.window_manager
        if wm is None:
            return True
        for w in wm.windows:
            for a in w.screen.areas:
                if a.type == "VIEW_3D":
                    sp = a.spaces.active
                    if sp is not None and sp.shading.type not in ("SOLID", "MATERIAL"):
                        return False
        return True

    def _tick(self):
        try:
            scene = bpy.context.scene
            if not self.active or scene is None or not hasattr(scene, "blendnite"):
                return 0.5
            if not scene.blendnite.enabled:
                return 0.5
            st = Settings.from_scene(scene)
            if st.build_sig() != self._last_sig:
                self._last_sig = st.build_sig()
                self.dirty = True
            if not self._pruned:
                self._pruned = True
                self._get_pool(WORKERS).submit(_job_prune, st.cache_dir)
            susp = not self._views_ok()
            if susp != self.suspended:
                self.suspended = susp
                self.dirty = True
            ao = bpy.context.object
            mode = (ao.name, ao.mode) if ao is not None else None
            if mode != self._last_mode:           # entering / leaving edit mode
                self._last_mode = mode
                self.dirty = True
            finished = self._poll()
            now = time.time()
            changed = finished
            if self.dirty or finished or now - self._last_full > 1.0:
                self.dirty = False
                self._last_full = now
                changed |= self.sync_scene(scene, st)
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
                                            thread_name_prefix="blendnite")
            self._pool_workers = workers
        return self._pool

    def _poll(self) -> bool:
        done = [k for k, f in self.futures.items() if f.done()]
        for k in done:
            f = self.futures.pop(k)
            kind = self.kinds.pop(k, "build")
            self.fb_key.pop(k, None)
            try:
                a = f.result()
            except Exception as e:
                if kind == "patch":
                    self.no_patch.add(k)            # fall back to a full build
                else:
                    self.errors[k] = repr(e)
                    traceback.print_exc()
                continue
            if a is None:
                if kind == "patch":
                    self.no_patch.add(k)
                elif kind == "load":                # corrupt / outdated disk entry
                    try:
                        os.remove(self.paths[k])
                    except OSError:
                        pass
            else:
                self.assets[k] = a
        if done:
            self.dirty = True
        return bool(done)

    def _request(self, key, obj, mesh, topo, st: Settings):
        """Make sure `key` is being produced.  Returns an older asset that may be drawn
        in the meantime (patch case) or None."""
        if key in self.futures:
            fb = self.fb_key.get(key)
            return self.assets.get(fb) if fb else None
        if key in self.errors:
            return None
        pool = self._get_pool(WORKERS)
        path = os.path.join(st.cache_dir, key + ".npz")
        self.paths[key] = path
        if os.path.exists(path):
            self.futures[key] = pool.submit(_job_load, key, path)
            self.kinds[key] = "load"
            return None

        old_key = self.obj_key.get(obj.name)
        old = self.assets.get(old_key) if old_key else None
        if (old is not None and key not in self.no_patch and old.topo == topo
                and old.max_verts == st.max_verts and old.key != key
                and len(mesh.vertices) == len(old.src_co)):
            co = _get(mesh.vertices, "co", np.float32, len(mesh.vertices) * 3).reshape(-1, 3)
            self.futures[key] = pool.submit(_job_patch, old, co, key, topo)
            self.kinds[key] = "patch"
            self.fb_key[key] = old_key
            return old

        data = extract_mesh(mesh)
        if data is None:
            self.errors[key] = "no triangles"
            return None
        self.futures[key] = pool.submit(_job_build, key, data, st.max_verts, path, topo)
        self.kinds[key] = "build"
        return None

    # ── scene scan ────────────────────────────────────────────────────────
    @staticmethod
    def _eligible(obj, min_verts) -> bool:
        if obj.type != "MESH" or obj.mode != "OBJECT" or not obj.visible_get():
            return False
        if not (obj.display_type in ("TEXTURED", "SOLID")
                or (MARK in obj and obj.display_type == "BOUNDS")):
            return False
        mesh = obj.data
        if mesh is None or len(mesh.vertices) < min_verts or len(mesh.polygons) == 0 or mesh.shape_keys:
            return False
        for m in obj.modifiers:             # evaluated meshes are not clustered
            if m.show_viewport:
                return False
        return True

    def _digests_for(self, mesh):
        uid = mesh.session_uid
        sizes = (len(mesh.vertices), len(mesh.loops))
        m = self._memo.get(uid)
        if m is None or m[2:] != sizes:
            m = mesh_digests(mesh) + sizes
            self._memo[uid] = m
        return m[0], m[1]

    def sync_scene(self, scene, st: Settings) -> bool:
        seen = set()
        changed = False
        now = time.time()
        force_mats = self._mat_dirty
        if force_mats:
            self._mat_dirty = False
            self.mat_epoch += 1
            changed = True
        hide = not self.suspended

        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            if self.suspended or not self._eligible(obj, st.min_verts):
                release_native(obj)
                continue
            mesh = obj.data
            topo, cohash = self._digests_for(mesh)
            key = f"v{clustering.FORMAT_VERSION}_{topo}_{cohash}_{st.max_verts}"
            asset = self.assets.get(key)
            if asset is None:
                fb = self._request(key, obj, mesh, topo, st)
                if fb is None:
                    release_native(obj)             # stay native until ready
                    continue
                old_key = self.obj_key[obj.name]    # patch pending: keep drawing old asset
                self.last_used[old_key] = now
                inst = self.instances.get(obj.name)
                if inst is None:
                    inst = Instance(obj.name, old_key, fb)
                    inst.set_matrix(obj.matrix_world)
                    self.instances[obj.name] = inst
                    changed = True
                asset = fb
            else:
                self.last_used[key] = now
                inst = self.instances.get(obj.name)
                if inst is None or inst.key != key:
                    inst = Instance(obj.name, key, asset)
                    inst.set_matrix(obj.matrix_world)
                    self.instances[obj.name] = inst
                    changed = True
                self.obj_key[obj.name] = key
            seen.add(obj.name)

            sig = _slot_sig(obj)
            if force_mats or inst.mat_sig != sig:
                inst.mat_sig = sig
                inst.mats = [material_info(s.material) for s in obj.material_slots]
                changed = True
            if hide:
                _hide_native(obj)

        gone = [n for n in self.instances if n not in seen]
        for n in gone:
            del self.instances[n]
        if gone or changed:
            self.version += 1
            changed = True

        # evict assets nobody has used for a while
        pinned = set(self.futures) | set(self.fb_key.values())
        for k in list(self.assets):
            if k not in pinned and now - self.last_used.get(k, now) > IDLE_EVICT_S:
                del self.assets[k]
                self.last_used.pop(k, None)
        return changed

    def status(self) -> dict:
        tris = sum(i.asset.tri_count for i in self.instances.values())
        return {
            "instances": len(self.instances),
            "assets": len({i.key for i in self.instances.values()}),
            "pending": len(self.futures),
            "patching": sum(1 for k in self.kinds.values() if k == "patch"),
            "errors": len(self.errors),
            "source_tris": tris,
        }
