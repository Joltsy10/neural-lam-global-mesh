import numpy as np
import plotly.graph_objects as go

from geometry.icosahedron import get_icosahedron
from geometry.subdivison import refine
from geometry.cartesion import cartesian_to_latlon


def plot_mesh(vertices, faces, title="Icosahedral Mesh"):
    """
    Plot the icosahedral mesh as a 3D triangulated surface.

    Args:
        vertices: (N, 3) array of Cartesian coordinates
        faces: (F, 3) array of vertex indices
        title: plot title
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.3,
            color="lightblue",
            flatshading=True,
        ),
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=2, color="darkblue"),
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        )
    )

    fig.show()


def plot_g2m_edges(grid_lat, grid_lon, mesh_vertices, src, dst,
                   max_edges=500, title="G2M Edges"):
    """
    Plot G2M edges connecting grid nodes to mesh nodes.
    Samples one representative edge per mesh node to guarantee
    coverage — every mesh node with at least one edge will appear.

    Args:
        grid_lat: (N_grid,) latitude of grid nodes
        grid_lon: (N_grid,) longitude of grid nodes
        mesh_vertices: (N_mesh, 3) mesh node coordinates
        src: (E,) grid node indices
        dst: (E,) mesh node indices
        max_edges: max number of mesh nodes to draw an edge for
        title: plot title
    """
    from geometry.cartesion import latlon_to_cartesian
    grid_xyz = latlon_to_cartesian(grid_lat, grid_lon)

    # Diagnostic printout
    unique_mesh_with_edges = len(set(dst))
    print(f"Total edges: {len(src)}")
    print(f"Unique mesh nodes with at least one edge: {unique_mesh_with_edges} / {len(mesh_vertices)}")
    isolated = set(range(len(mesh_vertices))) - set(dst)
    print(f"Mesh nodes with zero edges: {len(isolated)}")

    # One representative edge per mesh node (first grid neighbor found)
    seen_mesh = {}
    for g_idx, m_idx in zip(src, dst):
        if m_idx not in seen_mesh:
            seen_mesh[m_idx] = g_idx

    # Cap at max_edges for rendering performance
    selected = list(seen_mesh.items())[:max_edges]

    edge_x, edge_y, edge_z = [], [], []
    for m_idx, g_idx in selected:
        g = grid_xyz[g_idx]
        m = mesh_vertices[m_idx]
        edge_x += [g[0], m[0], None]
        edge_y += [g[1], m[1], None]
        edge_z += [g[2], m[2], None]

    fig = go.Figure(data=[
        # Grid nodes
        go.Scatter3d(
            x=grid_xyz[:, 0], y=grid_xyz[:, 1], z=grid_xyz[:, 2],
            mode="markers",
            marker=dict(size=2, color="green"),
            name="Grid nodes"
        ),
        # Mesh nodes
        go.Scatter3d(
            x=mesh_vertices[:, 0], y=mesh_vertices[:, 1], z=mesh_vertices[:, 2],
            mode="markers",
            marker=dict(size=3, color="red"),
            name="Mesh nodes"
        ),
        # Edges
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="gray", width=1),
            name=f"G2M edges (1 per mesh node, up to {max_edges})"
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube")
    )

    fig.show()


if __name__ == "__main__":
    vertices, faces = get_icosahedron()

    # Plot base icosahedron
    plot_mesh(vertices, faces, title="Base Icosahedron (Level 0)")

    # Plot level 2 refinement
    v2, f2 = refine(vertices, faces, 2)
    plot_mesh(v2, f2, title="Refined Mesh (Level 2)")

    # Plot level 4 refinement
    v4, f4 = refine(vertices, faces, 4)
    plot_mesh(v4, f4, title="Refined Mesh (Level 4)")

    from geometry.g2m import build_g2m_edges, angular_to_euclidean_radius

    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 355, 72)
    grid_lat, grid_lon = np.meshgrid(lat, lon)
    grid_lat = grid_lat.flatten()
    grid_lon = grid_lon.flatten()

    vertices, faces = get_icosahedron()
    vertices, faces = refine(vertices, faces, 4)
    radius = angular_to_euclidean_radius(7.5)
    src, dst = build_g2m_edges(grid_lat, grid_lon, vertices, radius)

    plot_g2m_edges(grid_lat, grid_lon, vertices, src, dst,
                   max_edges=2562, title="G2M Edges Level 4")