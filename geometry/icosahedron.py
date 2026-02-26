import numpy as np

phi = (1 + np.sqrt(5)) / 2

def get_icosahedron():
    """
    Returns the vertices and faces of a unit icosahedron
    
    Returns:
        vertices: (12, 3) array of xyz coordinates on a unit sphere
        faces: (20, 3) array of vertex indices
    """

    vertices = np.array([
        #(0, ±1, ±φ)
        [0, 1 , phi], [0, -1, phi],
        [0, -1, -phi], [0, 1, -phi],
        #(±1, ±φ, 0)
        [1, phi, 0], [-1, phi, 0],
        [-1, -phi, 0], [1, -phi, 0],
        #(±φ, 0, ±1)
        [ phi, 0,  1], [ phi, 0, -1],
        [-phi, 0,  1], [-phi, 0, -1],
    ], dtype=np.float64)

    # Normalize to unit sphere
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array([
    [0, 2, 8], [0, 8, 4], [0, 4, 6], [0, 6, 10], [0, 10, 2],
    [3, 1, 9], [3, 9, 5], [3, 5, 7], [3, 7, 11], [3, 11, 1],
    [2, 5, 8], [8, 9, 4], [4, 1, 6], [6, 11, 10], [10, 7, 2],
    [5, 2, 7], [9, 8, 5], [1, 4, 9], [11, 6, 1], [7, 10, 11],
    ], dtype=np.int32)

    faces = fix_orientation(vertices, faces)
    return vertices, faces

def fix_orientation(vertices, faces):
    """
    Ensures all faces have consistent outward-facing normals
    (counter-clockwise when viewed from outside the sphere).
    """
    fixed = []
    for v1, v2, v3 in faces:
        V1, V2, V3 = vertices[v1], vertices[v2], vertices[v3]
        normal = np.cross(V2 - V1, V3 - V1)
        # Use face centroid as reference for outward direction
        centroid = (V1 + V2 + V3) / 3
        if np.dot(normal, centroid) < 0:
            fixed.append([v1, v3, v2])  # flip
        else:
            fixed.append([v1, v2, v3])
    return np.array(fixed, dtype=np.int32)

if __name__ == "__main__":
    vertices, faces = get_icosahedron()
    
    # Every vertex should be on unit sphere
    norms = np.linalg.norm(vertices, axis=1)
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"All on unit sphere: {np.allclose(norms, 1.0)}")

    for v1, v2, v3 in faces:
        V1, V2, V3 = vertices[v1], vertices[v2], vertices[v3]
        normal = np.cross(V2 - V1, V3 - V1)
        centroid = (V1 + V2 + V3) / 3
        assert np.dot(normal, centroid) > 0, f"Face {v1},{v2},{v3} has inward normal"
    print("All face normals point outward: True")