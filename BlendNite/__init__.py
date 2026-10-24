"""
BlendNite - GPU clustered viewport for Blender.

Import a model -> clusters + LODs build on background threads -> duplicates reuse the
same cluster asset (content-hashed, object space) -> distance selects coarser LODs with
fewer, bigger clusters -> vertex edits patch only the clusters around the edit.
"""

bl_info = {
    "name": "BlendNite",
    "author": "BlendNite",
    "version": (1, 0, 1),
    "blender": (5, 2, 0),
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


def _safe_unregister_class(c):
    """Unregister `c`, or a stale class of the same name left by a previous load."""
    for cls in (c, getattr(bpy.types, c.__name__, None)):
        if cls is None:
            continue
        try:
            bpy.utils.unregister_class(cls)
            return
        except (RuntimeError, ValueError):
            pass


def register():
    # clean leftovers from a failed / replaced earlier registration (e.g. re-install
    # without restarting Blender) so register_class never sees a half-state
    for c in reversed(ui.CLASSES):
        _safe_unregister_class(c)
    for c in ui.CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.blendnite = bpy.props.PointerProperty(type=ui.BlendNiteProps)
    for h, lst in ((_on_depsgraph, bpy.app.handlers.depsgraph_update_post),
                   (_on_load, bpy.app.handlers.load_post)):
        if h not in lst:
            lst.append(h)


def unregister():
    try:
        state.stop()
    except Exception:
        import traceback
        traceback.print_exc()
    for h, lst in ((_on_depsgraph, bpy.app.handlers.depsgraph_update_post),
                   (_on_load, bpy.app.handlers.load_post)):
        if h in lst:
            lst.remove(h)
    for c in reversed(ui.CLASSES):          # panel first, then the property group
        _safe_unregister_class(c)
    if hasattr(bpy.types.Scene, "blendnite"):
        del bpy.types.Scene.blendnite


if __name__ == "__main__":
    register()
