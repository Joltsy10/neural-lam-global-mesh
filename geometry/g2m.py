import numpy as np
from scipy.spatial import cKDTree

from geometry.icosahedron import get_icosahedron
from geometry.cartesion import latlon_to_cartesian
from geometry.subdivison import refine

def build_g2m_edges(grid_lat, grid_lon, mesh_vertices, radius):
    """
    Build Grid-to-Mesh edges by connected each mesh node to all 
    grid nodes within a given radius.
    
    Args:
        grid_lat: (N_grid, ) latitude of ERA5 grid node in degrees
        grid_lon: (N_grid, ) longitude of ERA5 grid node in degrees
        mesh_vertices: (N_mesh, 3) Cartesian coordinates of mesh nodes
        radius: float, Euclidean distance threshold on unit sphere
        
    Returns: 
        src: (E, ) array of grid node indices (sources)
        dst: (E, ) array of grid node indices (desinations)
    """

    grid_xyz = latlon_to_cartesian(grid_lat, grid_lon)
    tree = cKDTree(grid_xyz)

    src = []
    dst = []

    for mesh_idx, mesh_node in enumerate(mesh_vertices):
        neighbour_indices = tree.query_ball_point(mesh_node, r=radius)
        for grid_idx in neighbour_indices:
            src.append(grid_idx)
            dst.append(mesh_idx)

    return np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32)

def angular_to_euclidean_radius(angle_deg):
    """
    Convert an angular distance in degrees to euclidean distance
    on unit sphere
    
    Args:
        angle_deg: angular distance in degrees
    
    Returns:
        float: Euclidean distance on unit sphere
    """

    return 2 * np.sin(np.deg2rad(angle_deg / 2))

if __name__ == "__main__":
    # Build a small ERA5-like grid at 5 degree spacing
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 355, 72)
    grid_lat, grid_lon = np.meshgrid(lat, lon)
    grid_lat = grid_lat.flatten()
    grid_lon = grid_lon.flatten()

    # Build mesh at level 3
    vertices, faces = get_icosahedron()
    vertices, faces = refine(vertices, faces, 3)

    # Radius covering roughly 1.5x the mesh spacing at level 3
    radius = angular_to_euclidean_radius(5.0)

    src, dst = build_g2m_edges(grid_lat, grid_lon, vertices, radius)

    print(f"Grid nodes: {len(grid_lat)}")
    print(f"Mesh nodes: {len(vertices)}")
    print(f"G2M edges: {len(src)}")
    print(f"Avg grid neighbors per mesh node: {len(src)/len(vertices):.1f}")
    print(f"Any mesh node with no edges: {len(set(range(len(vertices))) - set(dst)) > 0}")