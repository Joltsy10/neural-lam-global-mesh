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

    return vertices, faces

if __name__ == "__main__":
    vertices, faces = get_icosahedron()
    
    # Every vertex should be on unit sphere
    norms = np.linalg.norm(vertices, axis=1)
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"All on unit sphere: {np.allclose(norms, 1.0)}")