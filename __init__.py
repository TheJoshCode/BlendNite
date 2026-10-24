"""
GPU Clustered Viewport for Blender.

Import a model -> clusters + LODs build on background threads -> duplicates reuse the
same cluster asset (content-hashed, stored in object space) -> distance selects coarser
LODs with fewer, bigger clusters, like Nanite's cluster cut.
"""

bl_info = {
    "name": "GPU Clustered Viewport",
    "author": "Hermes Agent",
    "version": (2, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Cluster",
    "description": "Async clustered LOD viewport drawing with shared assets for duplicates",
    "category": "3D View",
    "support": "TESTING",
}

import bpy
from bpy.app.handlers import persistent

from . import state, ui


@persistent
def _on_depsgraph(scene, depsgraph):
    state.sync.on_depsgraph(depsgraph)


@persistent
def _on_load(_dummy):
    state.renderer.stop()
    state.sync.reset_session()
    try:
        enabled = bpy.context.scene.gpu_cluster.enabled
    except AttributeError:
        enabled = False
    if enabled:
        state.start()
    else:
        state.sync.active = False
        state.sync.restore_all()      # clean up markers left by a crash


def register():
    for c in ui.CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.gpu_cluster = bpy.props.PointerProperty(type=ui.GPUClusterProps)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    state.stop()
    for h, lst in ((_on_depsgraph, bpy.app.handlers.depsgraph_update_post),
                   (_on_load, bpy.app.handlers.load_post)):
        if h in lst:
            lst.remove(h)
    del bpy.types.Scene.gpu_cluster
    for c in reversed(ui.CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
