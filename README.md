# Neural-LAM Global Graph Geometry

A from-scratch implementation of the core geometry pipeline for global weather forecasting with graph neural networks. Built as part of GSoC 2026 preparation for [Neural-LAM Project 4](https://github.com/mllam/neural-lam).

This mini-project implements the full encode-process-decode graph construction pipeline: icosahedral mesh generation, G2M edges, M2G barycentric interpolation, and hierarchical level mappings, without any dependency on GraphCast utilities.

---

## What This Is

Neural-LAM's global forecasting approach represents the atmosphere on two distinct node sets:

- **Grid nodes** — ERA5 lat/lon data points, where input and output live
- **Mesh nodes** — icosahedral mesh nodes used internally by the GNN

The GNN pipeline is:

```
ERA5 grid → [G2M] → icosahedral mesh → [M2M] → icosahedral mesh → [M2G] → ERA5 grid
              encode      process (hierarchical)      decode
```

This repo implements the geometry that makes this pipeline possible.

---

## Structure

```
geometry/
    icosahedron.py   — base icosahedron: 12 vertices, 20 faces, unit sphere
    subdivision.py    — recursive subdivision with midpoint deduplication
    cartesian.py     — lat/lon ↔ 3D Cartesian conversions on unit sphere
    g2m.py           — Grid-to-Mesh edges via radius query (KDTree)
    m2g.py           — Mesh-to-Grid barycentric interpolation weights
    hierarchy.py     — parent-child links between consecutive mesh levels
visualize.py         — 3D interactive visualizations using Plotly
```

---

## Geometry Modules

### `icosahedron.py`
Constructs the base icosahedron from the golden ratio. Returns 12 vertices on the unit sphere and 20 triangular faces. All face normals verified to point outward.

### `subdivision.py`
Subdivides each triangular face into 4 smaller triangles by inserting edge midpoints. Midpoints are normalized back onto the unit sphere and deduplicated across shared edges via a cache keyed on sorted edge pairs, failure to deduplicate silently breaks adjacency structure. After N levels: `10×4^N + 2` vertices, `20×4^N` faces.

| Level | Vertices | Faces |
|-------|----------|-------|
| 0     | 12       | 20    |
| 1     | 42       | 80    |
| 2     | 162      | 320   |
| 3     | 642      | 1280  |
| 4     | 2562     | 5120  |

### `cartesian.py`
Converts between lat/lon degrees and 3D Cartesian coordinates on the unit sphere:

```
x = cos(lat) * cos(lon)
y = cos(lat) * sin(lon)
z = sin(lat)
```

Inverse via `arcsin(z)` and `arctan2(y, x)`. Round-trip verified for non-pole points.

### `g2m.py`
Builds Grid-to-Mesh edges. For each mesh node, queries a KDTree of grid node Cartesian coordinates for all grid nodes within a Euclidean radius `r`. Angular threshold θ converts to Euclidean via `r = 2*sin(θ/2)`. Every mesh node is verified to have at least one edge thus full sphere coverage required.

```python
radius = angular_to_euclidean_radius(7.5)  # degrees
src, dst = build_g2m_edges(grid_lat, grid_lon, mesh_vertices, radius)
# src: grid node indices, dst: mesh node indices
```

### `m2g.py`
Builds Mesh-to-Grid barycentric interpolation weights. For each ERA5 grid node P, finds the mesh triangle containing it using a face-centroid KDTree, then computes three weights via sub-triangle areas:

```
w1 = area(P, V2, V3) / area(V1, V2, V3)
w2 = area(V1, P,  V3) / area(V1, V2, V3)
w3 = area(V1, V2, P)  / area(V1, V2, V3)
```

Weights are clamped and renormalized. At inference, decoding is a fixed weighted sum, so there are no learned parameters:

```python
prediction[P] = w1*mesh[V1] + w2*mesh[V2] + w3*mesh[V3]
```

### `hierarchy.py`
Builds parent-child relationships between consecutive mesh levels for hierarchical GNN processing. Each fine-level node is assigned to its nearest coarse-level node via KDTree. Inherited vertices (present at both levels) map to distance ≈ 0. New midpoint vertices map to whichever coarse vertex is closer.

Outputs `children_to_parent` (N_fine,) and `parent_to_children` (list of lists) for each level transition. Inter-level edges are derived directly from `children_to_parent`.

---

## Visualization

```bash
python visualize.py
```

Produces six interactive 3D Plotly plots:

1. Base icosahedron (level 0)
2. Refined mesh (level 2)
3. Refined mesh (level 4)
4. G2M edges — grid nodes (green) connected to mesh nodes (red)
5. Hierarchy level 0→1 — fine nodes (blue) connected to coarse parents (orange)
6. Hierarchy level 3→4 — same at full resolution

---

## Requirements

```
numpy
scipy
plotly
```

Install with:

```bash
pip install numpy scipy plotly
```

---

## Running Individual Modules

Each geometry module has a `__main__` block with verification checks:

```bash
python -m geometry.icosahedron   # vertex/face counts, unit sphere, outward normals
python -m geometry.subdivision    # counts at each level
python -m geometry.cartesian     # round-trip lat/lon verification
python -m geometry.g2m           # edge counts, coverage check
python -m geometry.m2g           # weights sum to 1, non-negative
python -m geometry.hierarchy     # parent-child coverage both directions
```

---

## Relationship to Neural-LAM

This implementation is a standalone reference for the geometry described in [Oskarsson et al. (2024)](https://arxiv.org/abs/2309.17370) and implemented in Neural-LAM's `prob_model_global` branch using GraphCast utilities. The goal here was to understand and implement the geometry independently, without relying on GraphCast as a black box.

The algorithms are identical. The M2G barycentric decoder, G2M radius query, and hierarchical level mappings here correspond directly to `create_global_mesh.py` in the global branch.
