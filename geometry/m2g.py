import numpy as np
from scipy.spatial import cKDTree 

from geometry.icosahedron import get_icosahedron
from geometry.subdivison import refine
from geometry.cartesion import latlon_to_cartesian

def triangle_area(A, B, C):
    """
    Area of triangle ABC in 3D Cartesian space via cross product
    area = ||(B - A) x (C - A)|| / 2
    """

    return np.linalg.norm(np.cross(B - A, C - A)) / 2

def raw_barycentric(P, V1, V2, V3):
    """
    Compute raw barycentric weights without clamping.
    Used to test containment - all three must be >= 0 for P inside triangle
    
    Returns:
        w: (3, ) array [w1, w2, w3]
    """

    total = triangle_area(V1, V2, V3)
    w1 = triangle_area(P, V2, V3) / total
    w2 = triangle_area(V1, P,  V3) / total
    w3 = triangle_area(V1, V2, P)  / total
    return np.array([w1, w2, w3])

def build_m2g(grid_lat, grid_lon, mesh_vertices, mesh_faces):
    """
    For each ERA5 grid node, find the mesh triangle containing it
    and compute barycentric interpolation weights
    
    Uses face-centroid KDTree for efficient triangle lookup
    
    Args:
        grid_lat: (N_grid, ) latitude of ERA5 grid node in degrees
        grid_lon: (N_grid, ) longitude of ERA5 grid node in degrees
        mesh_Vertices: (N_mesh, 3) Cartesian coordinates of mesh nodes
        mesh_faces: (F, 3) Mesh triangle vertex indices
        
    Returns:
        triangle_verts: (N_grid, 3) int - vertex indices of containing triangles
        barycentric_weights: (N_grid, 3) float - interpolations weight, sum to 1
    """

    grid_xyz = latlon_to_cartesian(grid_lat, grid_lon)

    #Building KDTree on face centroids
    face_xyz = mesh_vertices[mesh_faces]
    centroids = face_xyz.mean(axis=1)
    centroids /= np.linalg.norm(centroids, axis = 1, keepdims= True)

    tree = cKDTree(centroids)

    k = 4
    _, near_face_indices = tree.query(grid_xyz, k=k)

    n_grid = len(grid_xyz)
    out_verts = np.zeros((n_grid, 3), dtype=np.int32)
    out_weights = np.zeros((n_grid, 3), dtype=np.float64)

    for i, P in enumerate(grid_xyz):
        best_face = None
        best_weights = None
        best_min_w = -np.inf

        for face_idx in near_face_indices[i]:
            v1, v2, v3 = mesh_faces[face_idx]
            V1 = mesh_vertices[v1]
            V2 = mesh_vertices[v2]
            V3 = mesh_vertices[v3]

            w = raw_barycentric(P, V1, V2, V3)
            min_w = w.min()

            if min_w > best_min_w:
                best_min_w   = min_w
                best_face    = face_idx
                best_weights = w

            # Non-negative minimum means P is inside this triangle — stop searching
            if min_w >= -1e-10:
                break

        # Clamp and renormalize before storing
        best_weights = np.clip(best_weights, 0, None)
        best_weights /= best_weights.sum()

        out_verts[i]   = mesh_faces[best_face]
        out_weights[i] = best_weights

    return out_verts, out_weights

if __name__ == "__main__":
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 355, 72)
    grid_lat, grid_lon = np.meshgrid(lat, lon)
    grid_lat = grid_lat.flatten()
    grid_lon = grid_lon.flatten()

    vertices, faces = get_icosahedron()
    vertices, faces = refine(vertices, faces, 4)

    tri_verts, bary_weights = build_m2g(grid_lat, grid_lon, vertices, faces)

    print(f"Grid nodes:  {len(grid_lat)}")
    print(f"Mesh nodes:  {len(vertices)}, Faces: {len(faces)}")
    print(f"Weights sum to 1:       {np.allclose(bary_weights.sum(axis=1), 1.0)}")
    print(f"All weights non-negative: {(bary_weights >= 0).all()}")
    print(f"Sample weights (first 3 grid nodes):")
    for i in range(3):
        print(f"  Grid {i}: verts={tri_verts[i]}, weights={bary_weights[i].round(4)}")