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

def build_m2g_edges(m2g_edge_index, mesh_vertices, grid_lat, grid_lon):
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

def build_g2m_edges(g2m_edge_index, mesh_vertices, grid_lat, grid_lon):
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