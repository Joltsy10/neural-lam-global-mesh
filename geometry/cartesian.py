import numpy as np


def latlon_to_cartesian(lat, lon):
    """
    Convert latitude/longitude coordinates to 3D Cartesian on unit sphere.

    Args:
        lat: latitude in degrees, shape (N,), range [-90, 90]
        lon: longitude in degrees, shape (N,), range [0, 360] or [-180, 180]

    Returns:
        xyz: (N, 3) array of Cartesian coordinates on unit sphere
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.stack([x, y, z], axis=-1)


def cartesian_to_latlon(xyz):
    """
    Convert 3D Cartesian coordinates on unit sphere to lat/lon.

    Args:
        xyz: (N, 3) array of Cartesian coordinates on unit sphere

    Returns:
        lat: (N,) latitude in degrees, range [-90, 90]
        lon: (N,) longitude in degrees, range [-180, 180]
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    lat = np.rad2deg(np.arcsin(z))
    lon = np.rad2deg(np.arctan2(y, x))

    return lat, lon


if __name__ == "__main__":
    # Test round trip: latlon -> cartesian -> latlon
    lat = np.array([0.0, 45.0, -45.0, 90.0, -90.0])
    lon = np.array([0.0, 90.0, 180.0, 0.0, 0.0])

    xyz = latlon_to_cartesian(lat, lon)

    # Verify all points on unit sphere
    norms = np.linalg.norm(xyz, axis=1)
    print(f"All on unit sphere: {np.allclose(norms, 1.0)}")

    # Round trip
    lat_back, lon_back = cartesian_to_latlon(xyz)
    print(f"Lat round trip: {np.allclose(lat, lat_back, atol=1e-10)}")
    # Poles have undefined longitude so skip them for lon check
    non_pole = np.abs(lat) < 89.9
    print(f"Lon round trip: {np.allclose(lon[non_pole], lon_back[non_pole], atol=1e-10)}")

    # Test with mesh vertices
    from icosahedron import get_icosahedron
    from subdivison import refine
    vertices, faces = get_icosahedron()
    vertices, faces = refine(vertices, faces, 3)
    lat_mesh, lon_mesh = cartesian_to_latlon(vertices)
    print(f"Mesh lat range: [{lat_mesh.min():.1f}, {lat_mesh.max():.1f}]")
    print(f"Mesh lon range: [{lon_mesh.min():.1f}, {lon_mesh.max():.1f}]")
