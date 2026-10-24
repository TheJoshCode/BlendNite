import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, EnumProperty

from . import state


def _on_enabled(self, context):
    if self.enabled:
        state.start()
    else:
        state.stop()
    state.sync.tag_redraw()


def _on_build_change(self, context):
    state.sync.dirty = True


class BlendNiteProps(bpy.types.PropertyGroup):
    enabled: BoolProperty(
        name="Enable", default=False, update=_on_enabled,
        description="Draw eligible meshes through the BlendNite clustered GPU path")
    precision_px: FloatProperty(
        name="Precision (px)", default=10.0, min=0.5, max=64.0, step=50, precision=1,
        description="Maximum on-screen geometric error in pixels. Lower = more detail, "
                    "higher = coarser LODs sooner")
    max_cluster_verts: IntProperty(
        name="Max Verts / Cluster", default=128, min=16, max=1024, update=_on_build_change,
        description="Hard upper bound of unique vertices in one cluster "
                    "(changing this rebuilds assets)")
    max_drawn_verts: IntProperty(
        name="Max Drawn Verts", default=6_000_000, min=10_000, soft_max=100_000_000,
        description="Vertex budget per frame. When the scene would exceed it, distant "
                    "objects are pushed to coarser LODs until it fits")
    view_mode: EnumProperty(
        name="View Mode",
        items=[
            ('SHADED', "Shaded", "Standard material and texture shading"),
            ('CLAY', "Clay", "Uniform clay material shading"),
            ('CLUSTERS', "Clusters", "Color-coded cluster visualization"),
            ('VERTEXES', "Vertexes", "Wireframe and vertex view"),
        ],
        default='SHADED',
        description="Viewport rendering mode"
    )


class BLENDNITE_PT_main(bpy.types.Panel):
    bl_label = "BlendNite"
    bl_idname = "BLENDNITE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BlendNite"

    def draw(self, context):
        p = context.scene.blendnite
        l = self.layout
        l.prop(p, "enabled", toggle=True, icon="MESH_DATA")

        col = l.column(align=True)
        col.prop(p, "precision_px")
        col.prop(p, "max_cluster_verts")
        col.prop(p, "max_drawn_verts")

        l.separator()
        row = l.row(align=True)
        row.prop(p, "view_mode", expand=True)
        l.separator()

        if not p.enabled:
            return
        st = state.sync.status()
        rs = state.renderer.stats
        box = l.box()
        if state.sync.suspended:
            box.label(text="Paused: a 3D view is in Wireframe/Rendered", icon="PAUSE")
        box.label(text=f"Objects {st['instances']}  |  Unique assets {st['assets']}")
        if st["pending"]:
            what = "Updating edit" if st["patching"] else "Building"
            box.label(text=f"{what}: {st['pending']} ...", icon="SORTTIME")
        if st["errors"]:
            box.label(text=f"Failed: {st['errors']} (see console)", icon="ERROR")
        if rs["total_tris"]:
            box.label(text=f"Drawn {rs['tris']:,} / {rs['total_tris']:,} tris")
            box.label(text=f"Verts {rs['verts']:,}" + ("  (budget-limited)" if rs["limited"] else ""))


CLASSES = (BlendNiteProps, BLENDNITE_PT_main)