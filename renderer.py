"""
GPU renderer: one shared vertex/index buffer per (asset, LOD), drawn once per instance.

Per frame (all vectorised over instances):
  1. object-level frustum test (bounding spheres)
  2. screen-space-error LOD selection with hysteresis -> farther objects use coarser
     LODs, i.e. fewer and bigger clusters
  3. for large objects only: per-cluster frustum + normal-cone culling on the CPU,
     producing a partial index buffer (cached until the view changes)
"""
from __future__ import annotations

import traceback
from typing import Dict

import bpy
import gpu
import numpy as np
from mathutils import Matrix

CULL_MIN_TRIS = 50_000       # per-cluster culling only pays off above this
LOD_TINTS = [(0.80, 0.80, 0.80), (0.30, 0.80, 1.00), (0.50, 1.00, 0.40),
             (1.00, 0.90, 0.20), (1.00, 0.50, 0.20), (1.00, 0.30, 0.30),
             (0.80, 0.30, 1.00), (0.50, 0.50, 1.00)]
MODES = {"SHADED": 0, "TEXTURED": 1, "LOD": 2, "CLUSTER": 3}

_VERT = """
void main() {
    vN = normalize(mat3(normalMat) * nor);
    vUV = aUV;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

_FRAG = """
void main() {
    vec3 n = normalize(vN);
    if (!gl_FrontFacing) n = -n;
    float l = 0.35 + 0.65 * max(dot(n, normalize(vec3(0.3, 0.5, 0.8))), 0.0);
    vec3 c = texture(diffuseTex, vUV).rgb * color.rgb * l;
%s
    fragColor = vec4(c, 1.0);
}
"""

_CLUSTER_COLOR = """
    if (mode == 3) {
        float h = fract(sin(float(gl_PrimitiveID / clusterSize) * 12.9898) * 43758.5453);
        c = (0.5 + 0.5 * cos(6.28318 * (h + vec3(0.0, 0.33, 0.67)))) * l;
    }
"""


def _make_shader(with_cluster_debug: bool):
    iface = gpu.types.GPUStageInterfaceInfo("gpuc_iface")
    iface.smooth("VEC3", "vN")
    iface.smooth("VEC2", "vUV")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "mvp")
    info.push_constant("MAT4", "normalMat")
    info.push_constant("VEC4", "color")
    info.push_constant("INT", "mode")
    info.push_constant("INT", "clusterSize")
    info.vertex_input(0, "diffuseTex")
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_in(1, "VEC3", "nor")
    info.vertex_in(2, "VEC2", "aUV")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "fragColor")
    info.vertex_source(_VERT)
    frag = _FRAG % (_CLUSTER_COLOR if with_cluster_debug else "")
    info.fragment_source(frag)
    return gpu.shader.create_from_info(info)


class _GLevel:
    __slots__ = ("vbo", "ibo", "batch", "gpu_tex")


class Renderer:
    def __init__(self, sync):
        self.sync = sync
        self.handle = None
        self.shader = None
        self.cluster_debug_ok = True
        self.gl = {}                       # (asset key, lod) -> _GLevel
        self._tex_cache: Dict[str, object] = {}   # image name -> GPUTexture
        self.stats = {"drawn": 0, "tris": 0, "total_tris": 0}
        self._ver = None
        self._frame = 0
        self._failed = False
        self.insts = []

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self):
        if self.handle is None:
            self.handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw, (), "WINDOW", "POST_VIEW")

    def stop(self):
        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, "WINDOW")
            self.handle = None
        self.gl.clear()
        self.shader = None
        self._ver = None

    # ── GPU resources ─────────────────────────────────────────────────────
    def _get_shader(self):
        if self.shader is None:
            try:
                self.shader = _make_shader(True)
            except Exception:
                self.cluster_debug_ok = False
                self.shader = _make_shader(False)
        return self.shader

    def _gpu_level(self, asset, li):
        k = (asset.key, li)
        g = self.gl.get(k)
        if g is not None:
            return g
        lvl = asset.levels[li]
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
        fmt.attr_add(id="nor", comp_type="F32", len=3, fetch_mode="FLOAT")
        fmt.attr_add(id="aUV", comp_type="F32", len=2, fetch_mode="FLOAT")
        vbo = gpu.types.GPUVertBuf(fmt, len(lvl.positions))
        vbo.attr_fill("pos", lvl.positions)
        vbo.attr_fill("nor", lvl.normals)
        vbo.attr_fill("aUV", lvl.uvs)
        ibo = gpu.types.GPUIndexBuf(type="TRIS", seq=lvl.tris)
        g = _GLevel()
        g.vbo = vbo
        g.ibo = ibo
        g.batch = gpu.types.GPUBatch(type="TRIS", buf=vbo, elem=ibo)
        g.gpu_tex = None
        self.gl[k] = g
        return g

    # ── per-instance arrays ───────────────────────────────────────────────
    def _rebuild(self):
        prev = {i.name: l for i, l in zip(self.insts, getattr(self, "lev", []))}
        self.insts = list(self.sync.instances.values())
        n = len(self.insts)
        self.cw = np.array([i.center_w for i in self.insts]).reshape(n, 3)
        self.rw = np.array([i.radius_w for i in self.insts])
        self.sc = np.array([i.scale for i in self.insts])
        lmax = max((len(i.asset.levels) for i in self.insts), default=1)
        E = np.full((n, lmax), np.inf)
        for k, i in enumerate(self.insts):
            e = i.asset.errors
            E[k, :len(e)] = e
        self.E = E
        self.lev = np.array([min(prev.get(i.name, 0), len(i.asset.levels) - 1)
                             for i in self.insts], dtype=np.int64)
        self._ver = (self.sync.version, self.sync.xver)

    # ── partial draw (per-cluster culling) ────────────────────────────────
    def _partial(self, inst, li, lvl, g, PV, planes, cam, fwd, persp):
        M = inst.matrix
        sig = (li, np.round(PV @ M, 4).tobytes())
        if inst.cull is not None and inst.cull[0] == sig:
            return inst.cull[1], inst.cull[2]
        Mi = np.linalg.inv(M)
        pl = planes[:, :3] @ M[:3, :3]
        pd = planes[:, 3] + planes[:, :3] @ M[:3, 3]
        nl = np.maximum(np.linalg.norm(pl, axis=1), 1e-20)
        pl, pd = pl / nl[:, None], pd / nl
        keep = np.all(lvl.centers @ pl.T + pd >= -lvl.radii[:, None], axis=1)
        if persp:
            cam_l = Mi[:3, :3] @ cam + Mi[:3, 3]
            v = lvl.centers - cam_l
            ln = np.linalg.norm(v, axis=1)
            back = np.einsum("ij,ij->i", v, lvl.cone_axis) >= lvl.cone_cutoff * ln + lvl.radii
        else:
            f = Mi[:3, :3] @ fwd
            f /= max(np.linalg.norm(f), 1e-20)
            back = lvl.cone_axis @ f >= lvl.cone_cutoff
        keep &= ~back
        if keep.all():
            res = (None, lvl.n_tris)
        elif not keep.any():
            res = (False, 0)
        else:
            tri_keep = np.repeat(keep, np.diff(lvl.cluster_start))
            idx = np.ascontiguousarray(lvl.tris[tri_keep], np.int32)
            ibo = gpu.types.GPUIndexBuf(type="TRIS", seq=idx)
            res = (gpu.types.GPUBatch(type="TRIS", buf=g.vbo, elem=ibo), len(idx))
        inst.cull = (sig, res[0], res[1])
        return res

    # ── texture cache ───────────────────────────────────────────────────────
    def _get_or_create_texture(self, image_name: str):
        tex = self._tex_cache.get(image_name)
        if tex is not None:
            return tex
        img = bpy.data.images.get(image_name)
        if img is None or not img.pixels:
            return None
        try:
            tex = gpu.types.GPUTexture(image.size, data=img.pixels.as_buffer())
        except Exception:
            return None
        self._tex_cache[image_name] = tex
        return tex

    def _bind_texture(self, shader, image_name: str):
        tex = self._get_or_create_texture(image_name)
        if tex is None:
            return
        try:
            gpu.state.texture_unit_set(0)
            tex.bind(0)
            shader.uniform_sampler("diffuseTex", 0)
        except Exception:
            pass

    # ── draw ──────────────────────────────────────────────────────────────
    def draw(self):
        try:
            self._draw()
        except Exception:
            if not self._failed:
                self._failed = True
                traceback.print_exc()

    def _draw(self):
        ctx = bpy.context
        scene = ctx.scene
        if scene is None or not scene.gpu_cluster.enabled:
            return
        sd, rd, region = ctx.space_data, ctx.region_data, ctx.region
        if rd is None or region is None or sd.shading.type in ("WIREFRAME", "RENDERED"):
            return
        sy = self.sync
        if not sy.instances:
            self.stats.update(drawn=0, tris=0, total_tris=0)
            return
        if self._ver != (sy.version, sy.xver):
            self._rebuild()
        n = len(self.insts)
        if n == 0:
            return
        p = scene.gpu_cluster

        P = np.array(rd.window_matrix, dtype=np.float64)
        V = np.array(rd.view_matrix, dtype=np.float64)
        PV = P @ V
        Vi = np.linalg.inv(V)
        cam, fwd = Vi[:3, 3], -Vi[:3, 2]
        persp = abs(P[3, 3]) < 0.5
        H = region.height

        planes = np.stack([PV[3] + PV[0], PV[3] - PV[0], PV[3] + PV[1],
                           PV[3] - PV[1], PV[3] + PV[2], PV[3] - PV[2]])
        planes /= np.maximum(np.linalg.norm(planes[:, :3], axis=1, keepdims=True), 1e-20)
        vis = np.all(self.cw @ planes[:, :3].T + planes[:, 3] >= -self.rw[:, None], axis=1)

        with np.errstate(invalid="ignore", over="ignore"):
            if persp:
                dist = np.maximum(np.linalg.norm(self.cw - cam, axis=1) - self.rw, 1e-3)
                ppu = 0.5 * H * P[1, 1] / dist
            else:
                ppu = np.full(n, 0.5 * H * P[1, 1])
            err_px = self.E * (self.sc * ppu)[:, None]
            thr = float(p.error_px)
            cand = np.maximum((err_px <= thr).sum(axis=1) - 1, 0)
            ar = np.arange(n)
            prev = self.lev
            ep, ec = err_px[ar, prev], err_px[ar, cand]
            keep = ((prev > cand) & (ep <= thr * 1.25)) | ((prev < cand) & (ec > thr * 0.8))
            self.lev = np.where(keep, prev, cand)

        shader = self._get_shader()
        shader.bind()
        mode = MODES.get(p.debug_mode, 0)
        if mode == 3 and not self.cluster_debug_ok:
            mode = 0
        shader.uniform_int("mode", mode)
        shader.uniform_int("clusterSize", int(max(1, p.cluster_size)))
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)

        use_cull = p.cluster_culling and mode != 3
        drawn = tris = 0
        for i in np.nonzero(vis)[0]:
            inst = self.insts[i]
            li = int(self.lev[i])
            lvl = inst.asset.levels[li]
            g = self._gpu_level(inst.asset, li)
            batch, nt = g.batch, lvl.n_tris
            if use_cull and nt >= CULL_MIN_TRIS:
                b, nt = self._partial(inst, li, lvl, g, PV, planes, cam, fwd, persp)
                if b is False:
                    continue
                if b is not None:
                    batch = b
            M = inst.matrix
            VM = V @ M
            nm = np.eye(4)
            nm[:3, :3] = np.linalg.inv(VM[:3, :3]).T

            # Texture binding
            if mode in (0, 1):
                obj = bpy.data.objects.get(inst.name)
                if obj and obj.data.materials:
                    mat = obj.data.materials[0]
                    if mat and mat.image:
                        self._bind_texture(shader, mat.image.name)

            if mode == 1:
                # TEXTURED: use material color tint
                obj = bpy.data.objects.get(inst.name)
                if obj and obj.data.materials and obj.data.materials[0]:
                    mc = obj.data.materials[0].diffuse_color
                    tint = (mc[0], mc[1], mc[2], 1.0)
                else:
                    tint = (1.0, 1.0, 1.0, 1.0)
            else:
                tint = LOD_TINTS[min(li, len(LOD_TINTS) - 1)] if mode == 2 else LOD_TINTS[0]

            shader.uniform_float("mvp", Matrix((PV @ M).tolist()))
            shader.uniform_float("normalMat", Matrix(nm.tolist()))
            shader.uniform_float("color", tint)
            batch.draw(shader)
            drawn += 1
            tris += nt

        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("NONE")
        self.stats.update(drawn=drawn, tris=tris, total_tris=sy.status()["source_tris"])

        self._frame += 1
        if self._frame % 240 == 0:          # free GPU buffers of evicted assets
            live = set(sy.assets)
            for k in [k for k in self.gl if k[0] not in live]:
                del self.gl[k]
