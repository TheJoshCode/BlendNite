"""
GPU renderer: one shared vertex buffer per (asset, LOD) and one index buffer per
material group, drawn once per instance per material.

Per frame (vectorised over instances):
  1. object-level frustum test (bounding spheres)
  2. screen-space-error LOD selection (precision in px) with hysteresis
  3. drawn-vertex budget: if the selection exceeds "Max Drawn Verts", one global error
     multiplier is binary-searched until it fits - the farthest / least visible objects
     coarsen first
  4. large objects only: per-cluster frustum (+ optional back-face cone) culling, giving
     a partial index buffer per material group, cached until the view changes
"""
from __future__ import annotations

import math
import traceback

import bpy
import gpu
import numpy as np
from mathutils import Matrix

CULL_MIN_TRIS = 50_000       # per-cluster culling only pays off above this
CULL_REBUILD_BELOW = 0.75    # only build a partial index buffer if it drops >25 %
DEFAULT_MATERIAL = ((0.8, 0.8, 0.8, 1.0), None)

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
    vec3 base = (useTex != 0) ? texture(diffuseTex, vUV).rgb : color.rgb;
    fragColor = vec4(base * l, 1.0);
}
"""


def _make_shader():
    iface = gpu.types.GPUStageInterfaceInfo("blendnite_iface")
    iface.smooth("VEC3", "vN")
    iface.smooth("VEC2", "vUV")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "mvp")
    info.push_constant("MAT4", "normalMat")
    info.push_constant("VEC4", "color")
    info.push_constant("INT", "useTex")
    info.sampler(0, "FLOAT_2D", "diffuseTex")
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_in(1, "VEC3", "nor")
    info.vertex_in(2, "VEC2", "aUV")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "fragColor")
    info.vertex_source(_VERT)
    info.fragment_source(_FRAG)
    return gpu.shader.create_from_info(info)


class _Group:
    __slots__ = ("slot", "batch", "t0", "t1", "c0", "c1")


class _GLevel:
    __slots__ = ("vbo", "groups")


class Renderer:
    def __init__(self, sync):
        self.sync = sync
        self.handle = None
        self.shader = None
        self.white = None
        self.gl = {}                       # (asset key, lod) -> _GLevel
        self._tex = {}                     # image name -> GPUTexture | None
        self._mat_epoch = -1
        self.stats = {"drawn": 0, "tris": 0, "verts": 0, "total_tris": 0, "limited": False}
        self._ver = None
        self._frame = 0
        self._last_err = None
        self.insts = []
        self.lev = np.zeros(0, np.int64)

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
        self._tex.clear()
        self.shader = None
        self.white = None
        self._ver = None
        self.insts = []

    # ── GPU resources ─────────────────────────────────────────────────────
    def _get_shader(self):
        if self.shader is None:
            self.shader = _make_shader()
        return self.shader

    def _white(self):
        if self.white is None:
            self.white = gpu.types.GPUTexture(
                (1, 1), format="RGBA8", data=gpu.types.Buffer("FLOAT", 4, [1.0, 1.0, 1.0, 1.0]))
        return self.white

    def _texture(self, image_name):
        if not image_name:
            return None
        if image_name in self._tex:
            return self._tex[image_name]
        tex = None
        img = bpy.data.images.get(image_name)
        if img is not None and img.size[0] > 0 and img.size[1] > 0:
            try:
                tex = gpu.texture.from_image(img)
            except Exception:
                tex = None
        self._tex[image_name] = tex
        return tex

    def _gpu_level(self, asset, li) -> _GLevel:
        k = (asset.key, li)
        g = self.gl.get(k)
        if g is not None:
            return g
        lvl = asset.levels[li]
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
        fmt.attr_add(id="nor", comp_type="F32", len=3, fetch_mode="FLOAT")
        fmt.attr_add(id="aUV", comp_type="F32", len=2, fetch_mode="FLOAT")
        vbo = gpu.types.GPUVertBuf(fmt, lvl.n_verts)
        vbo.attr_fill("pos", lvl.positions)
        vbo.attr_fill("nor", lvl.normals)
        vbo.attr_fill("aUV", lvl.uvs)
        g = _GLevel()
        g.vbo = vbo
        g.groups = []
        cs = lvl.cluster_start
        for slot, c0, c1 in lvl.material_groups():
            gr = _Group()
            gr.slot, gr.c0, gr.c1 = slot, c0, c1
            gr.t0, gr.t1 = int(cs[c0]), int(cs[c1])
            ibo = gpu.types.GPUIndexBuf(
                type="TRIS", seq=np.ascontiguousarray(lvl.tris[gr.t0:gr.t1], np.int32))
            gr.batch = gpu.types.GPUBatch(type="TRIS", buf=vbo, elem=ibo)
            g.groups.append(gr)
        self.gl[k] = g
        return g

    # ── per-instance arrays ───────────────────────────────────────────────
    def _rebuild(self):
        prev = {i.name: l for i, l in zip(self.insts, self.lev)}
        self.insts = list(self.sync.instances.values())
        n = len(self.insts)
        self.cw = np.array([i.center_w for i in self.insts]).reshape(n, 3)
        self.rw = np.array([i.radius_w for i in self.insts])
        self.sc = np.array([i.scale for i in self.insts])
        lmax = max((len(i.asset.levels) for i in self.insts), default=1)
        E = np.full((n, lmax), np.inf)
        VC = np.zeros((n, lmax), np.int64)
        for k, i in enumerate(self.insts):
            e = i.asset.errors
            E[k, :len(e)] = e
            VC[k, :len(e)] = i.asset.vert_counts
        self.E, self.VC = E, VC
        self.lev = np.array([min(prev.get(i.name, 0), len(i.asset.levels) - 1)
                             for i in self.insts], dtype=np.int64)
        self._ver = (self.sync.version, self.sync.xver)

    # ── per-cluster culling ───────────────────────────────────────────────
    def _cull(self, inst, li, lvl, g, PV, planes, cam, fwd, persp, backface):
        """List aligned with g.groups of (batch | None=full | False=skip, n_tris)."""
        M = inst.matrix
        sig = (li, backface, np.round(PV @ M, 4).tobytes())
        if inst.cull is not None and inst.cull[0] == sig:
            return inst.cull[1]
        Mi = np.linalg.inv(M)
        pl = planes[:, :3] @ M[:3, :3]
        pd = planes[:, 3] + planes[:, :3] @ M[:3, 3]
        nl = np.maximum(np.linalg.norm(pl, axis=1), 1e-20)
        pl, pd = pl / nl[:, None], pd / nl
        keep = np.all(lvl.centers @ pl.T + pd >= -lvl.radii[:, None], axis=1)
        if backface and not inst.flip:
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
        cs = lvl.cluster_start
        out = []
        for gr in g.groups:
            k = keep[gr.c0:gr.c1]
            n_full = gr.t1 - gr.t0
            if k.all() or k.mean() >= CULL_REBUILD_BELOW:
                out.append((None, n_full))
            elif not k.any():
                out.append((False, 0))
            else:
                tri_keep = np.repeat(k, np.diff(cs[gr.c0:gr.c1 + 1]))
                idx = np.ascontiguousarray(lvl.tris[gr.t0:gr.t1][tri_keep], np.int32)
                ibo = gpu.types.GPUIndexBuf(type="TRIS", seq=idx)
                out.append((gpu.types.GPUBatch(type="TRIS", buf=g.vbo, elem=ibo), len(idx)))
        inst.cull = (sig, out)
        return out

    # ── draw ──────────────────────────────────────────────────────────────
    def draw(self):
        try:
            self._draw()
        except Exception:
            msg = traceback.format_exc()
            if msg != self._last_err:          # print each distinct error once
                self._last_err = msg
                print(msg)

    def _select_lods(self, vis, err_px, thr, budget):
        """Pick a LOD per instance.  Returns (levels, limited)."""
        n = len(self.insts)
        sel = np.flatnonzero(vis)

        def cand_for(scale):
            return np.maximum((err_px <= thr * scale).sum(axis=1) - 1, 0)

        def total(c):
            return int(self.VC[sel, c[sel]].sum())

        cand = cand_for(1.0)
        if total(cand) <= budget:
            ar = np.arange(n)
            prev = self.lev
            ep, ec = err_px[ar, prev], err_px[ar, cand]
            # hysteresis only between ADJACENT levels, so big view changes snap at once
            keep = (((prev == cand + 1) & (ep <= thr * 1.25))
                    | ((prev == cand - 1) & (ec > thr * 0.8)))
            return np.where(keep, prev, cand), False
        hi = 1e6
        if total(cand_for(hi)) > budget:
            return cand_for(hi), True          # everything already at its coarsest
        lo = 1.0
        for _ in range(16):
            mid = math.sqrt(lo * hi)
            if total(cand_for(mid)) <= budget:
                hi = mid
            else:
                lo = mid
        return cand_for(hi), True

    def _draw(self):
        ctx = bpy.context
        scene = ctx.scene
        if scene is None or not hasattr(scene, "blendnite") or not scene.blendnite.enabled:
            return
        sd, rd, region = ctx.space_data, ctx.region_data, ctx.region
        sy = self.sync
        if (rd is None or region is None or sy.suspended
                or sd.shading.type not in ("SOLID", "MATERIAL")):
            return
        if not sy.instances:
            self.stats.update(drawn=0, tris=0, verts=0, total_tris=0, limited=False)
            return
        if self._ver != (sy.version, sy.xver):
            self._rebuild()
        if self._mat_epoch != sy.mat_epoch:
            self._mat_epoch = sy.mat_epoch
            self._tex.clear()
        n = len(self.insts)
        if n == 0:
            return
        p = scene.blendnite

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
            self.lev, limited = self._select_lods(
                vis, err_px, float(p.precision_px), int(p.max_drawn_verts))

        shader = self._get_shader()
        shader.bind()
        white = self._white()
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)

        # back-face cone culling only when the viewport itself culls back faces,
        # otherwise open meshes would lose their inside surfaces
        backface = bool(getattr(sd.shading, "show_backface_culling", False))
        use_cull = True
        drawn = tris = verts = 0
        for i in np.nonzero(vis)[0]:
            inst = self.insts[i]
            li = int(self.lev[i])
            lvl = inst.asset.levels[li]
            g = self._gpu_level(inst.asset, li)
            parts = None
            if use_cull and lvl.n_tris >= CULL_MIN_TRIS:
                parts = self._cull(inst, li, lvl, g, PV, planes, cam, fwd, persp, backface)

            M = inst.matrix
            VM = V @ M
            nm = np.eye(4)
            nm[:3, :3] = np.linalg.inv(VM[:3, :3]).T
            shader.uniform_float("mvp", Matrix((PV @ M).tolist()))
            shader.uniform_float("normalMat", Matrix(nm.tolist()))

            any_drawn = False
            for gi, gr in enumerate(g.groups):
                batch, nt = gr.batch, gr.t1 - gr.t0
                if parts is not None:
                    b, nt = parts[gi]
                    if b is False:
                        continue
                    if b is not None:
                        batch = b
                mats = inst.mats
                color, image = (mats[min(gr.slot, len(mats) - 1)] if mats else DEFAULT_MATERIAL)
                tex = self._texture(image)
                shader.uniform_float("color", color)
                shader.uniform_int("useTex", 1 if tex is not None else 0)
                shader.uniform_sampler("diffuseTex", tex if tex is not None else white)
                batch.draw(shader)
                tris += nt
                any_drawn = True
            if any_drawn:
                drawn += 1
                verts += lvl.n_verts

        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("NONE")
        self.stats.update(drawn=drawn, tris=tris, verts=verts,
                          total_tris=sy.status()["source_tris"], limited=limited)

        self._frame += 1
        if self._frame % 240 == 0:          # free GPU buffers of evicted assets
            live = {i.key for i in self.insts}
            for k in [k for k in self.gl if k[0] not in live]:
                del self.gl[k]
