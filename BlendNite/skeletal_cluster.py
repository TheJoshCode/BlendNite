"""Runtime support for meshes deformed by armature modifiers."""
from __future__ import annotations

import bpy
import numpy as np


def is_skeletal_mesh(obj: bpy.types.Object) -> bool:
    """Return True when the mesh has a viewport-visible armature modifier."""
    if obj is None or obj.type != "MESH":
        return False
    return any(
        mod.type == "ARMATURE"
        and mod.show_viewport
        and mod.object is not None
        and mod.object.type == "ARMATURE"
        for mod in obj.modifiers
    )


def evaluate_skeletal_mesh(obj: bpy.types.Object, depsgraph):
    """Return current evaluated object-space positions and normals."""
    if not is_skeletal_mesh(obj) or depsgraph is None:
        return None

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        return None

    try:
        vertex_count = len(mesh.vertices)
        if vertex_count == 0:
            return None

        positions = np.empty(vertex_count * 3, dtype=np.float32)
        normals = np.empty(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", positions)

        try:
            mesh.calc_normals_split()
            loop_normals = np.empty(len(mesh.loops) * 3, dtype=np.float32)
            mesh.loops.foreach_get("normal", loop_normals)
            loop_normals = loop_normals.reshape(-1, 3)

            vertex_indices = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", vertex_indices)
            normal_sums = np.zeros((vertex_count, 3), dtype=np.float64)
            normal_counts = np.zeros(vertex_count, dtype=np.int32)
            np.add.at(normal_sums, vertex_indices, loop_normals)
            np.add.at(normal_counts, vertex_indices, 1)

            valid = normal_counts > 0
            normal_sums[valid] /= normal_counts[valid, None]
            lengths = np.linalg.norm(normal_sums, axis=1, keepdims=True)
            normal_sums = np.divide(
                normal_sums,
                lengths,
                out=np.zeros_like(normal_sums),
                where=lengths > 1e-8,
            )
            normals[:] = normal_sums.reshape(-1)
        except Exception:
            mesh.calc_loop_triangles()
            triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", triangles)
            triangles = triangles.reshape(-1, 3)

            pos = positions.reshape(-1, 3)
            triangle_normals = np.cross(
                pos[triangles[:, 1]] - pos[triangles[:, 0]],
                pos[triangles[:, 2]] - pos[triangles[:, 0]],
            )
            normal_sums = np.zeros((vertex_count, 3), dtype=np.float64)
            np.add.at(normal_sums, triangles[:, 0], triangle_normals)
            np.add.at(normal_sums, triangles[:, 1], triangle_normals)
            np.add.at(normal_sums, triangles[:, 2], triangle_normals)
            lengths = np.linalg.norm(normal_sums, axis=1, keepdims=True)
            normal_sums = np.divide(
                normal_sums,
                lengths,
                out=np.zeros_like(normal_sums),
                where=lengths > 1e-8,
            )
            normals[:] = normal_sums.reshape(-1)

        return positions.reshape(-1, 3), normals.reshape(-1, 3)
    finally:
        evaluated.to_mesh_clear()


def deformed_level_arrays(asset, level_index, evaluated_positions, evaluated_normals):
    """Map evaluated source positions and normals onto one clustered LOD level."""
    level = asset.levels[level_index]

    if level_index == 0:
        source_indices = np.asarray(level.src, dtype=np.int32)
        if len(source_indices) != level.n_verts:
            source_indices = np.arange(level.n_verts, dtype=np.int32)
        if np.any(source_indices < 0) or np.any(source_indices >= len(evaluated_positions)):
            return level.positions, level.normals
        return evaluated_positions[source_indices], evaluated_normals[source_indices]

    base_level = asset.levels[0]
    source_map = np.asarray(level.vmap, dtype=np.int32)
    base_source_indices = np.asarray(base_level.src, dtype=np.int32)
    if len(source_map) != len(base_source_indices):
        return level.positions, level.normals
    if np.any(base_source_indices < 0) or np.any(base_source_indices >= len(evaluated_positions)):
        return level.positions, level.normals

    base_positions = evaluated_positions[base_source_indices]
    base_normals = evaluated_normals[base_source_indices]
    positions = np.empty((level.n_verts, 3), dtype=np.float32)
    normals = np.empty((level.n_verts, 3), dtype=np.float32)

    for vertex_index in range(level.n_verts):
        base_indices = np.flatnonzero(source_map == vertex_index)
        if len(base_indices) == 0:
            positions[vertex_index] = level.positions[vertex_index]
            normals[vertex_index] = level.normals[vertex_index]
            continue

        positions[vertex_index] = base_positions[base_indices].mean(axis=0)
        normal = base_normals[base_indices].mean(axis=0)
        normal_length = float(np.linalg.norm(normal))
        if normal_length > 1e-6:
            normals[vertex_index] = normal / normal_length
        else:
            normals[vertex_index] = level.normals[vertex_index]

    return positions, normals
