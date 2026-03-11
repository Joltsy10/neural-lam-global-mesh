import numpy as np
import torch
from pathlib import Path

from geometry.icosahedron import get_icosahedron
from geometry.subdivison import refine
from geometry.cartesion import latlon_to_cartesian, cartesian_to_latlon
from geometry.g2m import build_g2m_edges, angular_to_euclidean_radius
from geometry.m2g import build_m2g
from geometry.hierarchy import build_hierarchy, get_inter_level_edges

def build_m2m_edges(faces):
    edges = set()
    for v1, v2, v3 in faces:
        edges.add((min(v1,v2), max(v1,v2)))
        edges.add((min(v2,v3), max(v2,v3)))
        edges.add((min(v1,v3), max(v1,v3)))

    edges = np.array(list(edges), dtype=np.int32)

    src = np.concatenate([edges[:,0], edges[:,1]])
    dst = np.concatenate([edges[:,1], edges[:,0]])

    return np.stack([src, dst], axis=0)

def compute_edge_features(src_xyz, dst_xyz):
    """
    For each edge (src, dst) compute [great_circle_length, delta_east, delta_north]
    in the tangent plane at the source node.
    
    Args:
        src_xyz: (E, 3) Cartesian coordinates of source nodes
        dst_xyz: (E, 3) Cartesian coordinates of destination nodes
        
    Returns:
        features: (E, 3) edge features
    """
    # Great circle length = angle between unit vectors
    dot = np.clip(np.sum(src_xyz * dst_xyz, axis=1), -1.0, 1.0)
    length = np.arccos(dot)

    # Tangent plane basis vectors at each source node
    lat, lon = cartesian_to_latlon(src_xyz)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    east  = np.stack([-np.sin(lon_rad),
                       np.cos(lon_rad),
                       np.zeros_like(lon_rad)], axis=1)
    
    north = np.stack([-np.sin(lat_rad) * np.cos(lon_rad),
                      -np.sin(lat_rad) * np.sin(lon_rad),
                       np.cos(lat_rad)], axis=1)
    
    # Project displacement onto tangent plane
    diff = dst_xyz - src_xyz
    delta_east = np.sum(diff * east, axis=1)
    delta_north = np.sum(diff * north, axis=1)

    return np.stack([length, delta_east, delta_north], axis=1)

def build_m2m_features(m2m_edge_index, mesh_vertices):
    """
    Compute edge features for M2M edges
    
    Args:
        m2m_edge_index: (2, E) int array of M2M edge indices
        mesh_vertices: (N_mesh, 3) Cartesian coordinates of mesh nodes
        
    Returns:
        features: (E, 3) edge features [length, delta_east, delta_north]
    """
    src_xyz = mesh_vertices[m2m_edge_index[0]]
    dst_xyz = mesh_vertices[m2m_edge_index[1]]

    return compute_edge_features(src_xyz, dst_xyz)

def build_m2g_features(m2g_edge_index, mesh_vertices, grid_lat, grid_lon):
    """
    Args:
        m2g_edge_index: (2, E) row 0 is mesh indices, row 1 is grid indices
        mesh_vertices: (N_mesh, 3)
        grid_lat: (N_grid,)
        grid_lon: (N_grid,)
    """
    src_xyz = mesh_vertices[m2g_edge_index[0]]
    dst_xyz = latlon_to_cartesian(grid_lat[m2g_edge_index[1]],
                                  grid_lon[m2g_edge_index[1]])
    
    return compute_edge_features(src_xyz, dst_xyz)

def build_g2m_features(g2m_edge_index, mesh_vertices, grid_lat, grid_lon):
    """
    Args:
        g2m_edge_index: (2, E) row 0 is grid indices, row 1 is mesh indices
        mesh_vertices: (N_mesh, 3)
        grid_lat: (N_grid,)
        grid_lon: (N_grid,)
    """
    src_xyz = latlon_to_cartesian(grid_lat[g2m_edge_index[0]],
                                  grid_lon[g2m_edge_index[0]])
    dst_xyz = mesh_vertices[g2m_edge_index[1]]

    return compute_edge_features(src_xyz, dst_xyz)

def build_graph(mesh_level, grid_lat, grid_lon, output_dir, g2m_angle_deg=7.5):
    """
    Build non-hierarchical icosahedral graph and save .pt files.

    Args:
        mesh_level: int, icosahedral refinement level
        grid_lat: (N_grid,)
        grid_lon: (N_grid,)
        output_dir: str on Path, where to save .pt files
        g2m_angle_deg: float, angular radius for G2M edges in degrees
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build mesh ---
    base_verts, base_faces = get_icosahedron()
    mesh_vertices, mesh_faces = refine(base_verts, base_faces, mesh_level)
    print(f"Mesh level {mesh_level}: {len(mesh_vertices)} nodes, {len(mesh_faces)} faces")

    # --- M2M edges ---
    m2m_edge_index = build_m2m_edges(mesh_faces)
    m2m_features   = build_m2m_features(m2m_edge_index, mesh_vertices)
    print(f"M2M edges: {m2m_edge_index.shape[1]}")

     # --- G2M edges ---
    radius = angular_to_euclidean_radius(g2m_angle_deg)
    g2m_src, g2m_dst = build_g2m_edges(grid_lat, grid_lon, mesh_vertices, radius)
    g2m_edge_index = np.stack([g2m_src, g2m_dst], axis=0)
    g2m_features   = build_g2m_features(g2m_edge_index, mesh_vertices, grid_lat, grid_lon)
    print(f"G2M edges: {g2m_edge_index.shape[1]}")

    # --- M2G edges ---
    triangle_verts, _ = build_m2g(grid_lat, grid_lon, mesh_vertices, mesh_faces)
    N_grid = len(grid_lat)
    m2g_src = triangle_verts.flatten()                      # (3*N_grid,) mesh indices
    m2g_dst = np.repeat(np.arange(N_grid), 3)              # (3*N_grid,) grid indices
    m2g_edge_index = np.stack([m2g_src, m2g_dst], axis=0)
    m2g_features   = build_m2g_features(m2g_edge_index, mesh_vertices, grid_lat, grid_lon)
    print(f"M2G edges: {m2g_edge_index.shape[1]}")

    # --- Mesh node features: [lat, lon] ---
    mesh_lat, mesh_lon = cartesian_to_latlon(mesh_vertices)
    mesh_features = np.stack([mesh_lat, mesh_lon], axis=1)  # (N_mesh, 2)

    torch.save([torch.tensor(m2m_edge_index, dtype=torch.long)],
               output_dir / "m2m_edge_index.pt")
    torch.save([torch.tensor(m2m_features, dtype=torch.float32)],
               output_dir / "m2m_features.pt")
    torch.save([torch.tensor(mesh_features, dtype=torch.float32)],
               output_dir / "mesh_features.pt")

    torch.save(torch.tensor(g2m_edge_index, dtype=torch.long),
               output_dir / "g2m_edge_index.pt")
    torch.save(torch.tensor(g2m_features, dtype=torch.float32),
               output_dir / "g2m_features.pt")

    torch.save(torch.tensor(m2g_edge_index, dtype=torch.long),
               output_dir / "m2g_edge_index.pt")
    torch.save(torch.tensor(m2g_features, dtype=torch.float32),
               output_dir / "m2g_features.pt")

    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 355, 72)
    grid_lat, grid_lon = np.meshgrid(lat, lon)
    grid_lat = grid_lat.flatten()
    grid_lon = grid_lon.flatten()

    build_graph(
        mesh_level=3,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        output_dir="graph_output"
    )
