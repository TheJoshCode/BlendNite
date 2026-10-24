import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, StringProperty)

from . import state
from .scene_sync import Settings, default_cache_dir


def _on_enabled(self, context):
    if self.enabled:
        state.start()
    else:
        state.stop()
    state.sync.tag_redraw()


def _on_cluster_change(self, context):
    state.sync.dirty = True


class GPUClusterProps(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Enable", default=False, update=_on_enabled,
                          description="Draw eligible meshes through the clustered GPU path")
    error_px: FloatProperty(
        name="Max Error (px)", default=1.5, min=0.25, max=16.0,
        description="Screen-space error budget. Lower = more detail / smaller clusters, "
                    "higher = coarser LODs sooner")
    cluster_size: IntProperty(
        name="Cluster Triangles", default=128, min=32, max=512, update=_on_cluster_change,
        description="Triangles per cluster (changing this rebuilds assets)")
    lod_levels: IntProperty(
        name="LOD Levels", default=6, min=1, max=8, update=_on_cluster_change,
        description="Maximum LOD levels per asset (changing this rebuilds assets)")
    min_polys: IntProperty(
        name="Min Polygons", default=5000, min=100, soft_max=200000,
        update=_on_cluster_change,
        description="Meshes smaller than this keep Blender's native drawing")
    workers: IntProperty(name="Build Threads", default=2, min=1, max=16,
                         description="Background threads used to build clusters")
    hide_native: BoolProperty(
        name="Replace Native Draw", default=True, update=_on_cluster_change,
        description="Show managed objects as bounds in Blender's own pass so the full mesh "
                    "is not drawn twice (original display type is restored when disabled)")
    cluster_culling: BoolProperty(
        name="Cluster Culling", default=True,
        description="Per-cluster frustum + back-face culling for large objects")
    debug_mode: EnumProperty(
        name="View", default="SHADED",
        items=[("SHADED", "Shaded", ""),
               ("LOD", "LOD Levels", "Tint by LOD level"),
               ("CLUSTER", "Clusters", "Colour by cluster (disables cluster culling)")])
    cache_dir: StringProperty(name="Cache Folder", subtype="DIR_PATH", default="",
                              description="Empty = Blender config folder")


class GPUC_OT_rebuild(bpy.types.Operator):
    bl_idname = "gpu_cluster.rebuild"
    bl_label = "Rebuild All"
    bl_description = "Drop in-memory assets and rebuild (disk cache is reused)"

    purge_disk: BoolProperty(default=False)

    def execute(self, context):
        st = Settings.from_scene(context.scene)
        state.renderer.gl.clear()
        state.sync.drop_assets(st.cache_dir if self.purge_disk else None)
        return {"FINISHED"}


class GPUC_PT_main(bpy.types.Panel):
    bl_label = "GPU Clusters"
    bl_idname = "GPUC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Cluster"

    def draw(self, context):
        p = context.scene.gpu_cluster
        l = self.layout
        l.prop(p, "enabled", toggle=True, icon="MESH_DATA")

        st = state.sync.status()
        rs = state.renderer.stats
        box = l.box()
        box.label(text=f"Objects: {st['instances']}   Unique assets: {st['assets']}")
        if st["pending"]:
            box.label(text=f"Building: {st['pending']} ...", icon="SORTTIME")
        if st["errors"]:
            box.label(text=f"Failed: {st['errors']} (see console)", icon="ERROR")
        if rs["total_tris"]:
            box.label(text=f"Drawn {rs['tris']:,} / {rs['total_tris']:,} tris")

        col = l.column(align=True)
        col.prop(p, "error_px")
        col.prop(p, "debug_mode")
        col.prop(p, "cluster_culling")

        col = l.column(align=True)
        col.label(text="Build")
        col.prop(p, "cluster_size")
        col.prop(p, "lod_levels")
        col.prop(p, "min_polys")
        col.prop(p, "workers")
        col.prop(p, "hide_native")

        col = l.column(align=True)
        col.prop(p, "cache_dir")
        row = col.row(align=True)
        row.operator("gpu_cluster.rebuild", text="Rebuild").purge_disk = False
        row.operator("gpu_cluster.rebuild", text="Purge Disk", icon="TRASH").purge_disk = True


CLASSES = (GPUClusterProps, GPUC_OT_rebuild, GPUC_PT_main)
