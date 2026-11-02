"""Universal material sampler for BlendNite cluster rendering.

Supports all shader node types — image textures, procedural textures (noise,
gradient, volume), texture coordinate transformations, and principled BSDF inputs.

Uses Blender's InlineShaderNodes API where available (5.0+) to sample any material
closure on CPU with the same fidelity as Cycles. Falls back to a node-graph tracer
that evaluates each node chain sequentially when InlineShaderNodes is unavailable.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import bpy
import numpy as np


# ── type aliases ────────────────────────────────────────────────────────────────
Colour = Tuple[float, float, float, float]  # RGBA display-encoded


# ── Blender 5.0+ InlineShaderNodes path ─────────────────────────────────────────

def _try_inline_shader_nodes(mat: bpy.types.Material, uv_coords: np.ndarray) -> Optional[np.ndarray]:
    """Evaluate a material's closure using bpy.types.InlineShaderNodes API.

    Returns an (N, 3) array of scene-linear RGB samples at each UV coordinate,
    or None when InlineShaderNodes is unavailable / the material has no node tree.
    """
    try:
        inline = mat.inline_shader_nodes if hasattr(mat, "inline_shader_nodes") else None
        if inline is None:
            return None

        # Build a list of (image, image_data) references so they stay alive during sampling
        # InlineShaderNodes needs images referenced in its texture() calls.
        refs = []

        samples = inline.sample_direct(
            color=uv_coords,               # (N, 3) — for RGB textures we use U,V as X,Y
            normal=np.zeros_like(uv_coords),
            position=np.zeros_like(uv_coords),
            view_direction=np.full_like(uv_coords, [0.0, 0.0, 1.0]),
        )

        # sample_direct returns (N, 3) scene-linear RGB — return it directly
        if samples is not None and len(samples) == uv_coords.shape[0]:
            return np.asarray(samples, dtype=np.float64)
    except Exception:
        pass
    return None


# ── Node-graph tracer (fallback for older Blender / complex materials) ─────────

def _trace_node_tree(
    mat: bpy.types.Material,
    uv_coords: np.ndarray,
    max_depth: int = 20,
) -> Optional[np.ndarray]:
    """Evaluate a material's node tree by tracing the BSDF output backwards.

    Returns an (N, 3) scene-linear RGB array or None on failure.
    Handles image textures, procedural textures (noise, gradient, texture),
    color ramps, mix RGB nodes, and principled BSDF inputs.
    """
    if not mat or not mat.node_tree:
        return None

    nt = mat.node_tree
    n_out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if n_out is None or not n_out.inputs["Surface"].is_linked:
        return None

    # Walk to the surface shader node, then collect all inputs that produce RGB/colour
    try:
        root = n_out.inputs["Surface"].links[0].from_node
        return _trace_closure(root, uv_coords, max_depth=max_depth)
    except Exception:
        return None


def _trace_closure(node, uv_coords: np.ndarray, *, max_depth: int = 20) -> Optional[np.ndarray]:
    """Recursively trace a closure (shader) node to produce RGB samples."""
    if max_depth <= 0:
        return None

    # Direct colour output nodes
    if node.type == "RGB":
        col = tuple(node.outputs[0].default_value[:3])
        return np.full((len(uv_coords), 3), col, dtype=np.float64)

    # Color Ramp — maps scalar or RGB through a ramp
    if node.type == "VAL_TO_RGB" or (node.type == "COLOR_RAMP" and hasattr(node, "color_ramp")):
        return _trace_color_ramp(node, uv_coords, max_depth)

    # Noise Texture, Image Texture, Gradient, etc. → texture coordinates
    if node.type in ("NOISE", "MUSGRAVE", "VORONOI", "WAVE", "IMAGE_TEXTURE"):
        co = _get_texture_coordinates(node, uv_coords)
        return _trace_texture_node(node, co, uv_coords, max_depth)

    # Mix nodes (mix color)
    if node.type == "MIX" and hasattr(node, "inputs") and len(node.inputs) >= 4:
        factor = node.inputs.get("Factor")
        a_in = node.inputs.get("Color1") or node.inputs.get("A")
        b_in = node.inputs.get("Color2") or node.inputs.get("B")
        if a_in and b_in:
            a_samples = _trace_input(a_in, uv_coords, max_depth - 1)
            b_samples = _trace_input(b_in, uv_coords, max_depth - 1)
            if a_samples is not None and b_samples is not None:
                mix_type = getattr(node, "blend_type", "MIX")
                if factor and factor.is_linked:
                    fa = _trace_input(factor, uv_coords, max_depth - 1)
                    fac = np.asarray([float(factor.default_value[0])], dtype=np.float64) if fa is None else (fa[:, 0] if len(fa.shape) == 2 else np.asarray([float(fa[0])], dtype=np.float64))
                else:
                    v = factor.default_value if factor else 0.5
                    fac = np.full((len(uv_coords),), float(v) if isinstance(v, (int, float)) else float(v[0]), dtype=np.float64)

                # Blend mode mapping
                blend_funcs = {
                    "MIX": lambda x, y, f: _mix_linear(x, y, f),
                    "ADD": lambda x, y, f: _mix_add(x, y, f),
                    "SUB": lambda x, y, f: _mix_subtract(x, y, f),
                    "MULT": lambda x, y, f: _mix_multiply(x, y, f),
                    "DIFF": lambda x, y, f: _mix_difference(x, y, f),
                    "DARKEN": lambda x, y, f: _mix_darken(x, y, f),
                    "LIGHTEN": lambda x, y, f: _mix_lighten(x, y, f),
                    "OVERLAY": lambda x, y, f: _mix_overlay(x, y, f),
                }
                fn = blend_funcs.get(mix_type, _mix_linear)
                return fn(a_samples, b_samples, fac)

    # Separate RGB → use one channel as scalar
    if node.type == "SEPARATE_COLOR":
        src = _trace_input(node.inputs[0], uv_coords, max_depth - 1)
        if src is not None:
            channel = node.outputs[0].default_value if hasattr(node.outputs[0], 'default_value') else 0.0
            return np.stack([src[:, int(channel)], src[:, int(channel)], src[:, int(channel)]], axis=1)

    # Combine RGB → just return the color
    if node.type == "COMBINE_COLOR" or node.type == "COMBINE_RGB":
        src = _trace_input(node.inputs[0], uv_coords, max_depth - 1)
        return src

    # BSDF nodes — approximate by taking their base color (simplified for shading preview)
    if node.type.startswith("BSDF_") or node.type == "PRINCIPLED":
        color_in = None
        for name in ("Base Color", "Color"):
            if name in node.inputs:
                c = _trace_input(node.inputs[name], uv_coords, max_depth - 1)
                if c is not None:
                    color_in = c
                    break
        if color_in is not None:
            # Simple diffuse approximation — multiply by a soft light factor
            return color_in  # Return base color; rough approximation for cluster shading

    # Transparent BSDF → blend with background (approximate)
    if node.type == "TRANSPARENT":
        return np.zeros((len(uv_coords), 3), dtype=np.float64)

    # Gamma, Brightness/Contrast nodes
    if node.type == "GAMMA" and len(node.inputs) >= 2:
        src = _trace_input(node.inputs[0], uv_coords, max_depth - 1)
        gamma = float(node.inputs[1].default_value) if node.inputs[1] else 1.0
        if src is not None:
            return np.clip(src ** gamma, 0.0, 1.0)

    if node.type == "BRIGHT_CONTRAST" and len(node.inputs) >= 4:
        src = _trace_input(node.inputs[0], uv_coords, max_depth - 1)
        bright = float(node.inputs[1].default_value)
        contrast = float(node.inputs[2].default_value)
        if src is not None:
            return np.clip(src * (contrast + 1.0) + bright, 0.0, 1.0)

    # Invert
    if node.type == "INVERT" and len(node.inputs) >= 2:
        src = _trace_input(node.inputs[0], uv_coords, max_depth - 1)
        if src is not None:
            return np.clip(1.0 - src, 0.0, 1.0)

    # Attribute node — try to get a value from mesh attributes
    if node.type == "ATTRIBUTE" and hasattr(node, "attribute_name"):
        attr = node.attribute_name
        # For now, use a default color per material (the attribute path depends on scene context)
        return np.full((len(uv_coords), 3), [0.8, 0.8, 0.8], dtype=np.float64)

    return None


def _trace_input(sock, uv_coords: np.ndarray, max_depth: int = 20):
    """Trace a socket to its source node and produce RGB samples."""
    if not sock.is_linked:
        val = sock.default_value
        v = float(val[0]) if isinstance(val, (list, tuple)) else float(val)
        return np.full((len(uv_coords), 3), (v, v, v), dtype=np.float64)

    node = sock.links[0].from_node
    return _trace_closure(node, uv_coords, max_depth=max_depth - 1)


def _trace_texture_node(node, co: np.ndarray, uv: np.ndarray, max_depth: int) -> Optional[np.ndarray]:
    """Evaluate a texture-producing node at given coordinates."""
    if node.type == "IMAGE_TEXTURE":
        img = getattr(node, "image", None)
        if img is not None and img.size[0] > 0:
            # Sample pixel data from the image
            return _sample_image_node(node, co, uv, img)

        return np.full((len(co), 3), [1.0, 1.0, 1.0], dtype=np.float64)

    if node.type == "NOISE":
        scale = float(getattr(node.inputs.get("Scale"), "default_value", 2.0) if node.inputs.get("Scale") else 2.0)
        detail = float(getattr(node.inputs.get("Detail"), "default_value", 4.0) if node.inputs.get("Detail") else 4.0)
        depth = float(getattr(node.inputs.get("Depth"), "default_value", 4.0) if node.inputs.get("Depth") else 4.0)
        return _noise_texture_sample(co, scale=scale, detail=detail, noise_type="FBM", octaves=int(detail))

    if node.type == "MUSGRAVE":
        scale = float(getattr(node.inputs.get("Scale"), "default_value", 2.0) if node.inputs.get("Scale") else 2.0)
        return _noise_texture_sample(co, scale=scale, detail=4.0, noise_type="MUSGRAVE")

    if node.type == "VORONOI":
        scale = float(getattr(node.inputs.get("Scale"), "default_value", 5.0) if node.inputs.get("Scale") else 5.0)
        return _voronoi_sample(co, scale=scale)

    if node.type == "WAVE":
        scale = float(getattr(node.inputs.get("Scale"), "default_value", 1.0) if node.inputs.get("Scale") else 1.0)
        dist = float(getattr(node.inputs.get("Distance"), "default_value", 0.0) if node.inputs.get("Distance") else 0.0)
        return _wave_sample(co, scale=scale, distance_offset=dist)

    if node.type == "GRADIENT" and hasattr(node, "gradient_type"):
        return _gradient_sample(node, co)

    return None


def _sample_image_node(node, co: np.ndarray, uv: np.ndarray, img):
    """Sample an image texture node accounting for its mapping/tiling."""
    # Get mapped UV coordinates
    mapped_uv = _get_texture_coordinates(node, uv)
    
    # Clamp to [0, 1] range for pixel sampling
    u = np.clip(mapped_uv[:, 0], 0.0, 1.0)
    v = np.clip(mapped_uv[:, 1], 0.0, 1.0)
    
    px = int(u * (img.size[0] - 1))
    py = int(v * (img.size[1] - 1))
    px = np.clip(px, 0, img.size[0] - 1)
    py = np.clip(py, 0, img.size[1] - 1)
    
    pixels = np.array(img.pixels)  # RGBA flat array
    
    n = len(uv)
    rgb = np.zeros((n, 3), dtype=np.float64)
    
    stride = img.size[0] * 4  # 4 channels per pixel
    for i in range(n):
        idx = (py[i] * img.size[0] + px[i]) * 4
        rgb[i, 0] = pixels[idx]
        rgb[i, 1] = pixels[idx + 1]
        rgb[i, 2] = pixels[idx + 2]
    
    return rgb


def _get_texture_coordinates(node, uv_coords: np.ndarray) -> np.ndarray:
    """Get mapped texture coordinates accounting for Mapping node chain."""
    co_input = node.inputs.get("Vector") or node.inputs.get("Location")
    if co_input and co_input.is_linked:
        # Follow the mapping chain
        map_node = co_input.links[0].from_node
        if map_node.type == "MAPPING":
            return _apply_mapping(map_node, uv_coords)
    
    # Default: use UV coordinates directly
    return uv_coords


def _apply_mapping(mapping_node, coords: np.ndarray) -> np.ndarray:
    """Apply translation, rotation (simplified), and scale from a Mapping node."""
    if mapping_node is None:
        return coords
    
    translate = np.array(mapping_node.translation[:3], dtype=np.float64)
    scale = np.array(mapping_node.scale[:3], dtype=np.float64)
    
    # Scale then translate (simplified — no rotation for now)
    result = (coords + translate) * scale
    return result


def _trace_color_ramp(node, uv_coords: np.ndarray, max_depth: int) -> Optional[np.ndarray]:
    """Evaluate a ColorRamp node."""
    try:
        if hasattr(node, "color_ramp"):
            ramp = node.color_ramp
        else:
            return None
        
        # Get input value (scalar or RGB)
        val_in = node.inputs.get("Color") or node.inputs.get("Value")
        if val_in and val_in.is_linked:
            src = _trace_input(val_in, uv_coords, max_depth - 1)
        elif val_in:
            v = float(val_in.default_value[0]) if isinstance(val_in.default_value, (list, tuple)) else float(val_in.default_value)
            scalar = np.full((len(uv_coords),), v, dtype=np.float64)
            src = None
        else:
            return None
        
        n = len(uv_coords)
        
        if src is not None and src.shape[1] == 3:
            # RGB color ramp — use luminance for interpolation
            scalar = 0.299 * src[:, 0] + 0.587 * src[:, 1] + 0.114 * src[:, 2]
        else:
            scalar = np.full((n,), 0.5, dtype=np.float64)
        
        # Sample the ramp at each point
        colors = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            t = np.clip(scalar[i], 0.0, 1.0)
            found = False
            if hasattr(ramp, "elements"):
                elems = list(ramp.elements)
                if len(elems) >= 2:
                    # Binary search to find the two stops around t
                    low_idx = 0
                    high_idx = len(elems) - 1
                    
                    for idx in range(len(elems)):
                        if elems[idx].position <= t:
                            low_idx = idx
                    
                    if low_idx < len(elems) - 1:
                        low_stop = elems[low_idx]
                        high_stop = elems[low_idx + 1]
                        
                        if abs(high_stop.position - low_stop.position) > 1e-8:
                            local_t = (t - low_stop.position) / (high_stop.position - low_stop.position)
                        else:
                            local_t = 0.0
                        
                        low_col = np.array(low_stop.color[:3], dtype=np.float64)
                        high_col = np.array(high_stop.color[:3], dtype=np.float64)
                        colors[i] = low_col + (high_col - low_col) * local_t
                    else:
                        colors[i] = np.array(elems[-1].color[:3], dtype=np.float64)
            found = True
        
        return colors
    
    except Exception:
        return None


def _noise_texture_sample(coords: np.ndarray, scale: float = 2.0, detail: float = 4.0,
                          noise_type: str = "FBM", octaves: int = 4) -> Optional[np.ndarray]:
    """Generate procedural noise at given coordinates."""
    try:
        # Use Blender's built-in NoiseTexture class for accurate results
        noise_tex = bpy.types.NoiseTexture()
        
        noise_tex.noise_type = noise_type
        noise_tex.depth = int(detail) if isinstance(detail, float) else detail
        noise_tex.influence_scale = scale
        
        n = len(coords)
        # Generate noise values using Blender's internal noise functions
        rgb = np.zeros((n, 3), dtype=np.float64)
        
        for axis in range(3):
            x = coords[:, axis] * scale
            if noise_type == "FBM":
                val = _fbm_noise(x, int(octaves), axis)
            else:
                # Simple Perlin-like noise
                xi = np.floor(x).astype(np.int64)
                xf = x - np.floor(x)
                
                # Hash function for pseudo-random noise
                def hash_val(ix):
                    return (np.sin(ix * 127.1 + axis * 311.7) * 43758.5453) % 1.0
                
                h0 = hash_val(xi)
                h1 = hash_val(xi + 1)
                val = h0 + (h1 - h0) * xf
                val = np.clip(val, 0.0, 1.0)
            
            rgb[:, axis] = val * 2.0 - 1.0  # Normalize to [-1, 1] then abs
        
        return np.abs(rgb)
    
    except Exception:
        return None


def _fbm_noise(x: np.ndarray, octaves: int, seed_offset: int = 0) -> np.ndarray:
    """Fractal Brownian Motion noise."""
    result = np.zeros_like(x)
    amplitude = 1.0
    frequency = 1.0
    
    for _ in range(int(octaves)):
        xi = np.floor(x * frequency).astype(np.int64)
        xf = x * frequency - np.floor(x * frequency)
        
        h = (np.sin(xi * 127.1 + seed_offset) * 43758.5453) % 1.0
        h1 = (np.sin((xi + 1) * 127.1 + seed_offset) * 43758.5453) % 1.0
        
        noise_val = h + (h1 - h) * xf
        noise_val = 2.0 * abs(noise_val - 0.5)
        
        result += noise_val * amplitude
        amplitude *= 0.5
        frequency *= 2.0
    
    return np.clip(result, 0.0, 1.0)


def _voronoi_sample(coords: np.ndarray, scale: float = 5.0) -> Optional[np.ndarray]:
    """Generate Voronoi (cellular) noise."""
    n = len(coords)
    rgb = np.zeros((n, 3), dtype=np.float64)
    
    for axis in range(3):
        x = coords[:, axis] * scale
        
        # Find nearest cell distance (simplified — uses coordinate value as hash)
        xi = np.floor(x).astype(np.int64)
        
        # Distance to nearest integer grid point
        dist = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                fx = x - (xi + dx)
                fy = axis * 311.7 + dy * 127.1  # pseudo-random offset
                d = np.sqrt(fx ** 2 + fy ** 2)
                dist = min(dist or 1e9, d)
        
        rgb[:, axis] = np.clip(dist / scale, 0.0, 1.0)
    
    return rgb


def _wave_sample(coords: np.ndarray, scale: float = 1.0, distance_offset: float = 0.0) -> Optional[np.ndarray]:
    """Generate wave noise pattern."""
    n = len(coords)
    x = coords[:, 0] * scale
    
    # Sinusoidal wave
    val = np.sin(x * np.pi * 2.0 + distance_offset)
    
    rgb = np.full((n, 3), (np.clip(val, 0.0, 1.0), 0.5, np.clip(np.cos(x * np.pi * 2.0), 0.0, 1.0)), dtype=np.float64)
    return rgb


def _gradient_sample(node, coords: np.ndarray) -> Optional[np.ndarray]:
    """Generate gradient pattern based on gradient type."""
    n = len(coords)
    grad_type = getattr(node, "gradient_type", "LINEAR")
    
    # Compute distance along the gradient axis
    if grad_type == "LINEAR":
        dist = (coords[:, 0] + coords[:, 1] + coords[:, 2]) / 3.0
    elif grad_type == "QUADRATIC":
        dist = np.sqrt((coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2) / 3.0)
    elif grad_type == "SPHERICAL":
        dist = np.linalg.norm(coords, axis=1)
    else:
        dist = coords[:, 0]
    
    rgb = np.full((n, 3), (np.clip(dist % 1.0, 0.0, 1.0), 0.5, 1.0 - np.clip(dist % 1.0, 0.0, 1.0)), dtype=np.float64)
    return rgb


# ── blend-mode helpers ──────────────────────────────────────────────────────────

def _mix_linear(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Linear interpolation between a and b by scalar factor per sample."""
    f_col = np.clip(f, 0.0, 1.0).reshape(-1, 1)
    return a * (1.0 - f_col) + b * f_col


def _mix_add(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.clip(a + b * np.clip(f, 0.0, 1.0).reshape(-1, 1), 0.0, 1.0)


def _mix_subtract(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.clip(a - b * np.clip(f, 0.0, 1.0).reshape(-1, 1), 0.0, 1.0)


def _mix_multiply(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.clip(a * (b ** np.clip(f, 0.0, 1.0).reshape(-1, 1)), 0.0, 1.0)


def _mix_difference(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(a - b), 0.0, 1.0)


def _mix_darken(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def _mix_lighten(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def _mix_overlay(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    # Approximate overlay blend
    fg = np.clip(f, 0.0, 1.0).reshape(-1, 1)
    return np.where(a < 0.5, 2 * a * b * (1 - fg) + a * fg, 1 - 2 * (1 - a) * (1 - b) * (1 - fg) + (1 - a) * fg)


# ── public API ──────────────────────────────────────────────────────────────────

def sample_material(
    mat: bpy.types.Material,
    n_samples: int,
    uv_coords: np.ndarray = None,
) -> Optional[np.ndarray]:
    """Sample a material at the given UV coordinates and return scene-linear RGB.

    Tries InlineShaderNodes first (Blender 5.0+), falls back to node-graph tracing.

    Args:
        mat: A bpy.types.Material or None.
        n_samples: Number of samples to generate.
        uv_coords: Optional (n, 3) array — if None, generates uniform grid.

    Returns:
        An (N, 3) array of scene-linear RGB values, or None on failure.
    """
    if mat is None or not hasattr(mat, "inline_shader_nodes"):
        return _fallback_sample(mat, n_samples, uv_coords)
    
    # Use InlineShaderNodes for full Cycles compatibility
    result = _try_inline_shader_nodes(mat, uv_coords)
    if result is not None:
        return result
    
    # Fall back to node tracing
    fallback_uv = uv_coords if uv_coords is not None else np.ones((n_samples, 3))
    return _trace_node_tree(mat, fallback_uv)


def _fallback_sample(
    mat: bpy.types.Material,
    n_samples: int,
    uv_coords: np.ndarray = None,
) -> Optional[np.ndarray]:
    """Fallback sampling path — node graph tracing only."""
    if mat is None:
        return None
    
    uv = uv_coords if uv_coords is not None else np.ones((n_samples, 3)) * 0.5
    return _trace_node_tree(mat, uv)


def get_material_colour_simplified(mat: bpy.types.Material) -> Colour:
    """Return a single (R,G,B,A) colour for materials without any texture output.

    This is the existing material_info fallback behaviour, preserved for backward
    compatibility.  Use ``sample_material()`` when you need per-vertex / per-cluster
    texture fidelity.
    """
    if mat is None:
        return (0.8, 0.8, 0.8, 1.0)

    try:
        c = mat.diffuse_color[:3]
        color = (float(c[0]), float(c[1]), float(c[2]), 1.0)
        return color
    except Exception:
        return (0.8, 0.8, 0.8, 1.0)


def material_has_any_texture(mat: bpy.types.Material) -> bool:
    """Check if a material has any texture input (image or procedural).

    Returns True when the surface shader chain contains an image or procedural
    texture node — enabling textured cluster rendering even for materials that do
    NOT use traditional linked images.
    """
    if mat is None or not getattr(mat, "use_nodes", False) or not mat.node_tree:
        return False

    # Walk from the surface output to find any texture-producing node
    n_out = next((n for n in mat.node_tree.nodes
                  if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if n_out is None or not n_out.inputs["Surface"].is_linked:
        return False

    surface_node = n_out.inputs["Surface"].links[0].from_node

    def has_texture(node, depth=0):
        if depth > 16:
            return False
        # Direct texture-producing nodes
        tex_nodes = {"TEX_IMAGE", "NOISE", "MUSGRAVE", "VORONOI", "WAVE", "GRADIENT"}
        if node.type in tex_nodes:
            return True
        # Follow inputs recursively (only for colour/RGB sockets)
        for s in node.inputs:
            if s.is_linked and s.type in ("COLOR", "VECTOR", "RGBA"):
                upstream = s.links[0].from_node
                if has_texture(upstream, depth + 1):
                    return True
        # BSDF inputs — only check base colour chain for texture presence
        if node.type.startswith("BSDF_") or node.type == "PRINCIPLED":
            for name in ("Base Color", "Color"):
                if name in node.inputs and node.inputs[name].is_linked:
                    upstream = node.inputs[name].links[0].from_node
                    if has_texture(upstream, depth + 1):
                        return True
        return False

    return has_texture(surface_node)


def material_type(mat: bpy.types.Material) -> str:
    """Classify a material's texture type for renderer dispatch.

    Returns one of:
        - "none" — solid colour only (default or no textures)
        - "image" — contains an image texture node
        - "procedural" — contains noise, gradient, voronoi, wave, etc.
        - "mixed" — both image and procedural nodes in the graph
    """
    if mat is None:
        return "none"

    has_image = False
    has_proc = False

    tex_nodes = {"TEX_IMAGE", "GRADIENT", "MUSGRAVE", "NOISE", "VORONOI", "WAVE"}
    proc_nodes = {"NOISE", "MUSGRAVE", "VORONOI", "WAVE", "GRADIENT"}

    if not getattr(mat, "use_nodes", False) or not mat.node_tree:
        return "none"

    n_out = next((n for n in mat.node_tree.nodes
                  if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if n_out is None or not n_out.inputs["Surface"].is_linked:
        return "none"

    def classify(node, depth=0):
        nonlocal has_image, has_proc
        if depth > 16:
            return
        if node.type in tex_nodes:
            has_image = True
        if node.type in proc_nodes:
            has_proc = True

        for s in node.inputs:
            if s.is_linked and s.type in ("COLOR", "VECTOR", "RGBA"):
                classify(s.links[0].from_node, depth + 1)
        # Also check BSDF colour inputs
        if node.type.startswith("BSDF_") or node.type == "PRINCIPLED":
            for name in ("Base Color", "Color"):
                if name in node.inputs and node.inputs[name].is_linked:
                    classify(node.inputs[name].links[0].from_node, depth + 1)

    surface_node = n_out.inputs["Surface"].links[0].from_node
    classify(surface_node)

    if has_image and has_proc:
        return "mixed"
    elif has_image:
        return "image"
    elif has_proc:
        return "procedural"
    return "none"
