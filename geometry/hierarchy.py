import numpy as np
from scipy.spatial import cKDTree 

from geometry.icosahedron import get_icosahedron
from geometry.subdivison import refine

def build_hierarchy(base_vertices, base_faces, n_levels):
    """
    Build parent-child relationships between consecutive mesh levels.
    
    For each consecutive pair of levels (coarse, fine) computes:
    - children_to_parent: (N_fine, ) int - for each fine node, its parents
      coarse node index
    - parent_to_children: list of list - for each coarse node, which fine 
      node indices point to it
      
    Args:
        base_vertices: (12, 3) base icosahedron vertices
        base_faces: (20, 3) base icosahedron faces
        n_levels: int, number of refinement levels
        
    Returns:
        levels: list of (vertices, faces) tuples, one per level 0..n_levels
        mappings: list of dicts, one per consecutive level pair each dict
                  each dict has keys 'children_to_parent' and 
                  'parent_to_children'
    """

    # Build all mesh levels first
    levels = []
    v, f = base_vertices.copy(), base_faces.copy()
    levels.append((v, f))
    for _ in range(n_levels):
        v, f = refine(v, f, 1) # One level at a time
        levels.append((v, f))

    # For each consecutive pair, compute the mapping
    mappings = []
    for coarse_idx in range(len(levels) - 1):
        fine_idx = coarse_idx + 1
        coarse_verts = levels[coarse_idx][0] # (N_coarse, 3)
        fine_verts = levels[fine_idx][0] # (N_fine, 3)

        mapping = build_level_mapping(coarse_verts, fine_verts)
        mappings.append(mapping)

    return levels, mappings

def build_level_mapping(coarse_verts, fine_verts):
    """
    For one coarse->fine levle pair, find which fine node each coarse
    node corresponds to, then assign evey fine node to its nearest
    coarse node.
    
    Args:
        coarse_verts: (N_coarse, 3) Cartesian coordinates of coarse vertices
        fine_verts: (N_fine, 3) Cartesion coordinates of fine vertices
        
    Returns:
        dict with:
            'children_to_parent': (N_fine, ) int array
            'parent_to_children': list of (N_coarse, ) lists of int
    """
    n_coarse = len(coarse_verts)
    n_fine = len(fine_verts)

    # Build KDTree on coarse vertices
    # For each fine vertex, find nearest coarse vertex
    tree = cKDTree(coarse_verts)
    distances, parent_indices = tree.query(fine_verts, k=1)

    # parent_indices[i] = which coarse node fine node i maps to
    children_to_parent = parent_indices.astype(np.int32)

    # Invert: for each coarse node, collect all fine nodes assigned to it
    parent_to_children = [[] for _ in range(n_coarse)]
    for fine_node, coarse_node in enumerate(children_to_parent):
        parent_to_children[coarse_node].append(fine_node)

    # Sanity: every coarse node should have atleast one child
    empty = [i for i, children in enumerate(parent_to_children) 
             if len(children) ==0]
    if empty:
        raise ValueError(f"Coarse nodes with no parent: {empty}")
    
    return {
        'children_to_parent': children_to_parent,
        'parent_to_children': parent_to_children
    }

def get_inter_level_edges(mapping):
    """
    Convert a level mapping into edge arrays for the GNN.
    Inter-level edges connect each fine node to its parent coarse node.
    
    Args:
        mapping: dict from build_level_mapping
        
    Returns:
        src: (N_fine, ) fine node indices (upward: source is fine)
        dst: (N_fine, ) coarse node indices (upward: dest is coarse)
    """

    children_to_parent = mapping['children_to_parent']
    n_fine = len(children_to_parent)

    src = np.arange(n_fine, dtype=np.int32) # every fine node
    dst = children_to_parent                # its parent

    return src, dst

if __name__ == "__main__":
    vertices, faces = get_icosahedron()
    n_levels = 4

    levels, mappings = build_hierarchy(vertices, faces, n_levels)

    print(f"Mesh levels built: {len(levels)}")
    for i, (v, f) in enumerate(levels):
        print(f"  Level {i}: {len(v)} vertices, {len(f)} faces")

    print(f"\nLevel transition mappings: {len(mappings)}")
    for i, mapping in enumerate(mappings):
        c2p = mapping['children_to_parent']
        p2c = mapping['parent_to_children']
        n_coarse = len(p2c)
        n_fine   = len(c2p)
        avg_children = np.mean([len(c) for c in p2c])
        print(f"  Level {i}→{i+1}: {n_coarse} coarse nodes, "
              f"{n_fine} fine nodes, "
              f"avg children per coarse node: {avg_children:.1f}")

    print(f"\nInter-level edges (Level 0→1):")
    src, dst = get_inter_level_edges(mappings[0])
    print(f"  src (fine) range:   [{src.min()}, {src.max()}]")
    print(f"  dst (coarse) range: [{dst.min()}, {dst.max()}]")
    print(f"  Total edges: {len(src)}")
    print(f"  Every fine node has a parent: "
          f"{len(set(src)) == len(levels[1][0])}")
    print(f"  Every coarse node has children: "
          f"{len(set(dst)) == len(levels[0][0])}")