"""Process-wide singletons (kept out of __init__ to avoid circular imports)."""
from .scene_sync import SceneSync
from .renderer import Renderer
from . import debug as bn_debug

sync = SceneSync()
renderer = Renderer(sync)


def start():
    bn_debug.info("State", "Starting BlendNite")
    sync.activate()
    renderer.start()


def stop():
    bn_debug.info("State", "Stopping BlendNite")
    renderer.stop()
    sync.deactivate()
