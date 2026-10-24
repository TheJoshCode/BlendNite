"""
BlendNite - GPU clustered viewport for Blender.

Import a model -> clusters + LODs build on background threads -> duplicates reuse the
same cluster asset (content-hashed, object space) -> distance selects coarser LODs with
fewer, bigger clusters -> vertex edits patch only the clusters around the edit.
"""

bl_info = {
    "name": "BlendNite",
    "author": "BlendNite",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > BlendNite",
    "description": "Async clustered LOD viewport drawing with shared assets, material "
                   "support and incremental edit updates",
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
        enabled = bpy.context.scene.blendnite.enabled
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
    bpy.types.Scene.blendnite = bpy.props.PointerProperty(type=ui.BlendNiteProps)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    state.stop()
    for h, lst in ((_on_depsgraph, bpy.app.handlers.depsgraph_update_post),
                   (_on_load, bpy.app.handlers.load_post)):
        if h in lst:
            lst.remove(h)
    del bpy.types.Scene.blendnite
    for c in reversed(ui.CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
