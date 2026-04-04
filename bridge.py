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
    dot = np.clip(np.sum(src_xyz * dst_xyz, axis=1), -1.0, 1.0)
    length = np.arccos(dot)

    lat, lon = cartesian_to_latlon(src_xyz)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    east  = np.stack([-np.sin(lon_rad),
                       np.cos(lon_rad),
                       np.zeros_like(lon_rad)], axis=1)

    north = np.stack([-np.sin(lat_rad) * np.cos(lon_rad),
                      -np.sin(lat_rad) * np.sin(lon_rad),
                       np.cos(lat_rad)], axis=1)

    diff = dst_xyz - src_xyz
    delta_east  = np.sum(diff * east,  axis=1)
    delta_north = np.sum(diff * north, axis=1)

    return np.stack([length, delta_east, delta_north], axis=1)

def build_m2m_features(m2m_edge_index, mesh_vertices):
    src_xyz = mesh_vertices[m2m_edge_index[0]]
    dst_xyz = mesh_vertices[m2m_edge_index[1]]
    return compute_edge_features(src_xyz, dst_xyz)

def build_m2g_features(m2g_edge_index, mesh_vertices, grid_lat, grid_lon):
    src_xyz = mesh_vertices[m2g_edge_index[0]]
    dst_xyz = latlon_to_cartesian(grid_lat[m2g_edge_index[1]],
                                  grid_lon[m2g_edge_index[1]])
    return compute_edge_features(src_xyz, dst_xyz)

def build_g2m_features(g2m_edge_index, mesh_vertices, grid_lat, grid_lon):
    src_xyz = latlon_to_cartesian(grid_lat[g2m_edge_index[0]],
                                  grid_lon[g2m_edge_index[0]])
    dst_xyz = mesh_vertices[g2m_edge_index[1]]
    return compute_edge_features(src_xyz, dst_xyz)

def build_inter_level_features(edge_index, coarse_vertices, fine_vertices):
    src_xyz = fine_vertices[edge_index[0]]
    dst_xyz = fine_vertices[edge_index[1]]
    return compute_edge_features(src_xyz, dst_xyz)

def build_graph(mesh_level, grid_lat, grid_lon, g2m_angle_deg=7.5):
    """
    Build hierarchical icosahedral graph and return all data as a dict.

    Args:
        mesh_level: int, number of icosahedral refinement levels
        grid_lat: (N_grid,)
        grid_lon: (N_grid,)
        g2m_angle_deg: float, angular radius for G2M edges in degrees

    Returns:
        dict with all graph arrays
    """
    base_verts, base_faces = get_icosahedron()
    levels, mappings = build_hierarchy(base_verts, base_faces, mesh_level)

    print(f"Mesh levels built: {len(levels)}")
    for i, (v, f) in enumerate(levels):
        print(f"  Level {i}: {len(v)} nodes, {len(f)} faces")

    m2m_edge_index_list = []
    m2m_features_list   = []
    for i, (verts, faces) in enumerate(levels):
        ei = build_m2m_edges(faces)
        ft = build_m2m_features(ei, verts)
        m2m_edge_index_list.append(ei)
        m2m_features_list.append(ft)
        print(f"  M2M level {i}: {ei.shape[1]} edges")

    mesh_lat_lon_list = []
    for verts, _ in levels:
        lat, lon = cartesian_to_latlon(verts)
        mesh_lat_lon_list.append(np.stack([lat, lon], axis=1))

    finest_verts = levels[-1][0]
    radius = angular_to_euclidean_radius(g2m_angle_deg)
    g2m_src, g2m_dst = build_g2m_edges(grid_lat, grid_lon, finest_verts, radius)
    g2m_edge_index = np.stack([g2m_src, g2m_dst], axis=0)
    g2m_features   = build_g2m_features(g2m_edge_index, finest_verts, grid_lat, grid_lon)
    print(f"G2M edges: {g2m_edge_index.shape[1]}")

    finest_faces = levels[-1][1]
    triangle_verts, _ = build_m2g(grid_lat, grid_lon, finest_verts, finest_faces)
    N_grid = len(grid_lat)
    m2g_src = triangle_verts.flatten()
    m2g_dst = np.repeat(np.arange(N_grid), 3)
    m2g_edge_index = np.stack([m2g_src, m2g_dst], axis=0)
    m2g_features   = build_m2g_features(m2g_edge_index, finest_verts, grid_lat, grid_lon)
    print(f"M2G edges: {m2g_edge_index.shape[1]}")

    mesh_up_edge_index_list   = []
    mesh_up_features_list     = []
    mesh_down_edge_index_list = []
    mesh_down_features_list   = []

    for i, mapping in enumerate(mappings):
        coarse_verts = levels[i][0]
        fine_verts   = levels[i+1][0]

        up_src, up_dst = get_inter_level_edges(mapping)
        up_ei = np.stack([up_src, up_dst], axis=0)
        up_ft = build_inter_level_features(up_ei, coarse_verts, fine_verts)

        down_ei = np.stack([up_dst, up_src], axis=0)
        down_src_xyz = coarse_verts[down_ei[0]]
        down_dst_xyz = fine_verts[down_ei[1]]
        down_ft = compute_edge_features(down_src_xyz, down_dst_xyz)

        mesh_up_edge_index_list.append(up_ei)
        mesh_up_features_list.append(up_ft)
        mesh_down_edge_index_list.append(down_ei)
        mesh_down_features_list.append(down_ft)

        print(f"  Up/down edges level {i}->{i+1}: {up_ei.shape[1]}")

    return {
        'm2m_edge_index': m2m_edge_index_list,
        'm2m_features':   m2m_features_list,
        'mesh_lat_lon':   mesh_lat_lon_list,
        'g2m_edge_index': g2m_edge_index,
        'g2m_features':   g2m_features,
        'm2g_edge_index': m2g_edge_index,
        'm2g_features':   m2g_features,
        'up_edge_index':  mesh_up_edge_index_list,
        'up_features':    mesh_up_features_list,
        'down_edge_index':mesh_down_edge_index_list,
        'down_features':  mesh_down_features_list,
    }


if __name__ == "__main__":
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 355, 72)
    grid_lat, grid_lon = np.meshgrid(lat, lon)
    grid_lat = grid_lat.flatten()
    grid_lon = grid_lon.flatten()

    data = build_graph(
        mesh_level=3,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
    )
    print("Keys:", list(data.keys()))