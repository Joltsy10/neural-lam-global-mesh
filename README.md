# Neural-LAM Global Mesh

A from-scratch implementation of the icosahedral graph geometry pipeline for global weather forecasting with GNNs, plus a bridge layer that produces files in the exact format `utils.load_graph` expects in [Neural-LAM](https://github.com/mllam/neural-lam).

Built as part of GSoC 2026 preparation for [Neural-LAM Project 4](https://github.com/mllam/neural-lam). The companion training repo is [gnn-weather-from-scratch](https://github.com/Joltsy10/gnn-weather-from-scratch).

---

## What This Is

Neural-LAM's global forecasting approach represents the atmosphere on two distinct node sets:

- **Grid nodes** — ERA5 lat/lon data points, where inputs and outputs live
- **Mesh nodes** — icosahedral mesh nodes used internally by the GNN processor

The encode-process-decode pipeline is:

```
ERA5 grid → [G2M] → icosahedral mesh → [M2M hierarchical] → icosahedral mesh → [M2G] → ERA5 grid
              encode         process (up-down sweep)              decode
```

This repo implements the geometry that makes this pipeline possible, and `bridge.py` converts it into `.pt` files ready for neural-lam.

---

## Structure

```
geometry/
    icosahedron.py    — base icosahedron: 12 vertices, 20 faces, unit sphere
    subdivision.py    — recursive subdivision with midpoint deduplication
    cartesian.py      — lat/lon ↔ 3D Cartesian conversions on unit sphere
    g2m.py            — Grid-to-Mesh edges via angular radius query (KDTree)
    m2g.py            — Mesh-to-Grid barycentric interpolation
    hierarchy.py      — parent-child links between consecutive mesh levels
bridge.py             — converts geometry outputs to neural-lam .pt format
visualize.py          — 3D interactive visualizations using Plotly
```

---

## Geometry Modules

### `icosahedron.py`
Constructs the base icosahedron from the golden ratio. Returns 12 vertices on the unit sphere and 20 triangular faces. All face normals verified to point outward.

### `subdivision.py`
Subdivides each triangular face into 4 smaller triangles by inserting edge midpoints. Midpoints are normalized back onto the unit sphere and deduplicated across shared edges via a cache keyed on sorted edge pairs. Failure to deduplicate silently breaks adjacency structure.

After N subdivision levels: `10×4^N + 2` vertices, `20×4^N` faces.

| Level | Vertices | Faces | Approx. spacing |
|-------|----------|-------|-----------------|
| 0 | 12 | 20 | ~63° |
| 1 | 42 | 80 | ~33° |
| 2 | 162 | 320 | ~16° |
| 3 | 642 | 1280 | ~8° |
| 4 | 2562 | 5120 | ~4° |

### `cartesian.py`
Converts between lat/lon degrees and 3D Cartesian coordinates on the unit sphere:

```
x = cos(lat) * cos(lon)
y = cos(lat) * sin(lon)
z = sin(lat)
```

Inverse via `arcsin(z)` and `arctan2(y, x)`. Round-trip verified for non-pole points.

### `g2m.py`
Builds Grid-to-Mesh edges. For each mesh node, queries a KDTree of grid node Cartesian coordinates for all grid nodes within angular radius θ (default 7.5°). Angular threshold converts to Euclidean via `r = 2*sin(θ/2)`. Every mesh node is verified to have at least one edge.

### `m2g.py`
Builds Mesh-to-Grid edges via triangle containment. For each ERA5 grid node P, finds the enclosing mesh triangle using a face-centroid KDTree, then computes barycentric weights via sub-triangle areas:

```
w1 = area(P, V2, V3) / area(V1, V2, V3)
w2 = area(V1, P,  V3) / area(V1, V2, V3)
w3 = area(V1, V2, P)  / area(V1, V2, V3)
```

Each grid node connects to exactly 3 mesh nodes. Weights are clamped and renormalized to sum to 1.

### `hierarchy.py`
Builds parent-child relationships between consecutive mesh levels for the hierarchical processor up-down sweep. Each fine-level node is assigned to its nearest coarse-level node via KDTree. Outputs `children_to_parent` (N_fine,) for each level transition, from which inter-level up and down edges are derived.

---

## Bridge Layer

`bridge.py` is the main deliverable of this repo. It takes the geometry outputs and produces all `.pt` files in the exact format `utils.load_graph` expects in neural-lam.

### Output files

**Non-hierarchical:**
- `g2m_edge_index.pt`, `g2m_features.pt`
- `m2g_edge_index.pt`, `m2g_features.pt`
- `m2m_edge_index.pt`, `m2m_features.pt` (lists of length `n_levels`)
- `mesh_features.pt` (list of length `n_levels`)

**Hierarchical additions:**
- `mesh_up_edge_index.pt`, `mesh_up_features.pt` (lists of length `n_levels - 1`)
- `mesh_down_edge_index.pt`, `mesh_down_features.pt` (lists of length `n_levels - 1`)

### Edge features

Edge features follow the tangential plane projection convention for spherical geometry: `[great_circle_length, Δx_tangential, Δy_tangential]` where the tangential plane is defined at the source node.

- East basis: `[-sin(lon), cos(lon), 0]`
- North basis: `[-sin(lat)cos(lon), -sin(lat)sin(lon), cos(lat)]`
- Great-circle length: `arccos(clip(dot(A,B), -1, 1))` — raw unnormalized, `load_graph` normalizes at load time

### Usage

```python
from bridge import build_graph

build_graph(
    mesh_level=3,
    grid_lat=lat_flat,       # (N_grid,) in degrees
    grid_lon=lon_flat,       # (N_grid,) in degrees
    output_dir='data/global',
    g2m_angle_deg=7.5
)
```

### Verified output (mesh level 3, 1° global grid)

| Component | Count |
|---|---|
| Grid nodes | 65,160 |
| Mesh nodes (finest) | 642 |
| M2M edges (finest level) | 3,840 |
| G2M edges | 177,160 |
| M2G edges | 195,480 (65,160 × 3) |

---

## Visualization

```bash
python visualize.py
```

Produces interactive 3D Plotly plots:

1. Base icosahedron (level 0)
2. Refined mesh (level 2)
3. Refined mesh (level 4)
4. G2M edges — grid nodes (green) connected to mesh nodes (red)
5. Hierarchy level 0→1 — fine nodes (blue) connected to coarse parents (orange)
6. Hierarchy level 3→4 — same at full resolution

---

## Running Individual Modules

Each geometry module has a `__main__` block with verification checks:

```bash
python -m geometry.icosahedron   # vertex/face counts, unit sphere, outward normals
python -m geometry.subdivision   # counts at each level
python -m geometry.cartesian     # round-trip lat/lon verification
python -m geometry.g2m           # edge counts, full coverage check
python -m geometry.m2g           # weights sum to 1, non-negative
python -m geometry.hierarchy     # parent-child coverage both directions
```

---

## Requirements

```bash
pip install numpy scipy plotly torch
```

---

## Relationship to Neural-LAM

The `.pt` files produced by `bridge.py` slot directly into neural-lam's `utils.load_graph` without any changes to the model code. `BaseHiGraphModel` reads `num_levels` dynamically from `len(self.mesh_static_features)`, so a hierarchical icosahedral graph at any `mesh_level` loads without architectural changes.

The edge feature convention (`[length, Δx_tangential, Δy_tangential]`) differs semantically from the rectilinear LAM convention (`[length, Δx, Δy]`) but `EDGE_FEATURE_DIM = 3` stays correct in both cases. This is one of the spec gaps tracked in [neural-lam#348](https://github.com/mllam/neural-lam/issues/348).