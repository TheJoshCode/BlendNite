"""Central logging for BlendNite console diagnostics."""
from __future__ import annotations

import threading
import time

enabled = True
_printed = {}
_lock = threading.Lock()


def set_enabled(value: bool) -> None:
    global enabled
    enabled = bool(value)


def info(component: str, message: str) -> None:
    if enabled:
        print(f"[BlendNite] [INFO] [{component}] {message}")


def debug(component: str, message: str) -> None:
    if enabled:
        print(f"[BlendNite] [DEBUG] [{component}] {message}")


def warn(component: str, message: str) -> None:
    if enabled:
        print(f"[BlendNite] [WARN] [{component}] {message}")


def debug_throttled(component: str, key: str, message: str, interval_s: float = 1.0) -> None:
    if not enabled:
        return
    now = time.monotonic()
    token = (component, key)
    with _lock:
        previous = _printed.get(token, 0.0)
        if now - previous < interval_s:
            return
        _printed[token] = now
    print(f"[BlendNite] [DEBUG] [{component}] {message}")

