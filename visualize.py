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