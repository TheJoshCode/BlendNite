"""Process-wide singletons (kept out of __init__ to avoid circular imports)."""
from .scene_sync import SceneSync
from .renderer import Renderer

sync = SceneSync()
renderer = Renderer(sync)


def start():
    sync.activate()
    renderer.start()


def stop():
    renderer.stop()
    sync.deactivate()
