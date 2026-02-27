import numpy as np
from icosahedron import get_icosahedron

def subdivide(vertices, faces):
    """
    Subdivides each triangular face into 4 smaller triangles
    Midpoints are deduplicated across shared edges
    
    Args:
        vertices: (N, 3) array of xyz coordinates on a unit sphere
        faces: (F, 3) array of vertex indices
        
    Returns:
        new_vertices: (M, 3) array of xyz coordinates on a unit sphere
        new_faces: (F*3, 3) array of vertex indices
    """

    vertices = list(vertices)
    new_faces = []
    midpoint_cache = {}

    def get_midpoint(i, j):
        key = tuple(sorted((i, j)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        
        midpoint = (vertices[i] + vertices[j]) / 2
        midpoint = midpoint / np.linalg.norm(midpoint)

        index = len(vertices)
        vertices.append(midpoint)
        midpoint_cache[key] = index
        return index
    
    for v1, v2, v3 in faces:
        m12 = get_midpoint(v1, v2)
        m23 = get_midpoint(v2, v3)
        m13 = get_midpoint(v1, v3)

        new_faces.append([v1, m12, m13])
        new_faces.append([v2, m23, m12])
        new_faces.append([v3, m13, m23])
        new_faces.append([m12, m23, m13])

    return np.array(vertices, dtype=np.float64), np.array(new_faces, dtype=np.int32)

def refine(vertices, faces, levels):
    """
    Applies subdivision repeatedly for a given number of levels
    
    Args:
        vertices: base icosahedron vertices
        faces: base icosahedron faces
        levels: number of subdivision
        
    Returns:
        vertices: vertices after subdividing
        faces: faces after subdividing
    """

    for _ in range(levels):
        vertices, faces = subdivide(vertices, faces)

    return vertices, faces

if __name__ == "__main__":
    vertices, faces = get_icosahedron()

    for level in range(1, 5):
        v, f = refine(vertices, faces, level)
        norms = np.linalg.norm(v, axis=1)
        expected_vertices = 10 * (4 ** level) + 2
        expected_faces = 20 * (4 ** level)
        print(f"Level {level}: {len(v)} vertices (expected {expected_vertices}), "
              f"{len(f)} faces (expected {expected_faces}), "
              f"all on sphere: {np.allclose(norms, 1.0)}")

    