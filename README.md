# GPU Clustered Viewport (Blender)

Import a model, clusters and LODs build on background threads. Duplicate it and the copy reuses the same cluster asset, with no rebuild. Farther objects switch to coarser LODs, which means fewer and bigger clusters.

## Install
Edit > Preferences > Add-ons > Install... and pick `blender_gpu_cluster.zip`, then enable it.
N-panel > **Cluster** tab > **Enable**.

## How it works
- **Async build**: mesh arrays are read on the main thread, then clustering, LOD generation and disk caching run in a thread pool. Objects stay natively drawn until their asset is ready.
- **Shared assets**: assets are keyed by a content hash (vertex positions plus topology) and stored in object space. Alt+D, Shift+D, and re-importing the same file all hit the same asset. Each instance only carries a matrix.
- **Disk cache**: one `.npz` per unique geometry in Blender's config folder, so reopening a scene skips the build.
- **Clusters**: triangles are Morton-sorted and chunked into 128-triangle clusters. Each cluster has a bounding sphere and a normal cone.
- **LOD**: grid vertex clustering at halving resolution gives about 4x fewer triangles per level. Each level has a known object-space error. LOD is chosen per object from projected pixel error (`Max Error (px)`), with hysteresis.
- **Textures**: the first material slot's image is bound per-instance and sampled in the fragment shader. UVs are averaged per-vertex during grid clustering. Switch to **Textured** view in the N-panel to enable.
- **Culling**: object-level frustum test is vectorised over all instances. Objects with 50k+ triangles also get per-cluster frustum and back-face culling.
- **Replace native draw**: managed objects switch to Bounds display so Blender does not draw them twice. The original display type is saved on the object and restored on disable.

## Limits
- The unit test covers the numpy core only. The Blender/GPU side has not been run inside Blender, so expect to fix API details on your Blender version.
- LOD is chosen per object, not per cluster, because grid simplification is not crack-free across mixed levels. A true Nanite DAG with locked cluster borders is the next step.
- Culling runs on the CPU with numpy. Blender's Python API has no indirect or instanced draw, so each visible instance is one Python draw call.
- Objects with viewport modifiers, shape keys, or in edit mode are left to native drawing.

## Test
`python tests/test_clustering.py` (needs only numpy).
