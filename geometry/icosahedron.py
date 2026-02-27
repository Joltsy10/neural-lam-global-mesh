import numpy as np

phi = (1 + np.sqrt(5)) / 2

def get_icosahedron():
    """
    Returns the vertices and faces of a unit icosahedron
    
    Returns:
        vertices: (12, 3) array of xyz coordinates on a unit sphere
        faces: (20, 3) array of vertex indices
    """
    vertices = []
    for c1 in [1., -1.]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.))
            vertices.append((0., c1, c2))
            vertices.append((c2, 0., c1))

    # Normalize to unit sphere
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array([[0, 1, 2],
           [0, 6, 1],
           [8, 0, 2],
           [8, 4, 0],
           [3, 8, 2],
           [3, 2, 7],
           [7, 2, 1],
           [0, 4, 6],
           [4, 11, 6],
           [6, 11, 5],
           [1, 5, 7],
           [4, 10, 11],
           [4, 8, 10],
           [10, 8, 3],
           [10, 3, 9],
           [11, 10, 9],
           [11, 9, 5],
           [5, 9, 7],
           [9, 3, 7],
           [1, 6, 5],
           ])

    return vertices, faces

if __name__ == "__main__":
    vertices, faces = get_icosahedron()
    
    # Every vertex should be on unit sphere
    norms = np.linalg.norm(vertices, axis=1)
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"All on unit sphere: {np.allclose(norms, 1.0)}")

    # Verify all face normals point outward using centroid
    all_correct = True
    for v1, v2, v3 in faces:
        V1, V2, V3 = vertices[v1], vertices[v2], vertices[v3]
        normal = np.cross(V2 - V1, V3 - V1)
        centroid = (V1 + V2 + V3) / 3
        if np.dot(normal, centroid) <= 0:
            all_correct = False
            print(f"Bad face: {v1},{v2},{v3}")
    
    print(f"All face normals point outward: {all_correct}")
    print(f"Expected 12 vertices, 20 faces: {len(vertices) == 12 and len(faces) == 20}")
