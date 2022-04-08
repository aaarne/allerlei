import numpy as np
import scipy.spatial
from .lines import ParametricCurve
from .tools import all_combinations


def triangulate_shape(point_chain, n, overlap_epsilon=0.5, boundary_points_factor=4):
    """
    Triangulate space enclosed by point_chain in 2D. The point chain must not intersect itself
    :param point_chain: n x 2 numpy array containing hull
    :param n: target number of points
    :return: points: N x 2 array of points (N is usually a bit more than n)
    :return: faces: F x 3 array of faces
    """
    from shapely.geometry import Polygon, Point
    boundary = Polygon(point_chain)
    n_boundary_points = boundary_points_factor * np.sqrt(n)

    # Create regular grid containing target polygon
    n_interior = n - n_boundary_points
    xmin, ymin, xmax, ymax = boundary.bounds
    too_many_points = all_combinations(
        np.linspace(xmin, xmax, 1 + int(np.ceil((xmax - xmin) * np.sqrt(n_interior / boundary.area)))),
        np.linspace(ymin, ymax, 1 + int(np.ceil((ymax - ymin) * np.sqrt(n_interior / boundary.area))))
    )

    # add boundary points of target polygon to the grid and remove points outside the polygon
    def check_points():
        for p in too_many_points:
            yield boundary.contains(Point(p[0], p[1]))

    is_interior = np.array([*check_points()])
    interior_points = too_many_points[is_interior]
    points = np.vstack((
        interior_points,
        ParametricCurve(point_chain).resample(int(n - np.sum(is_interior)))
    ))

    indices_of_boundary = np.arange(interior_points.shape[0], points.shape[0])

    # Delaunay-triangulate the points
    tri = scipy.spatial.Delaunay(points)
    faces = tri.simplices

    # remove faces outside the polygon
    def check_faces():
        for face in faces:
            face = Polygon([points[face[0]], points[face[1]], points[face[2]]])
            if face.area < 1e-9:
                yield False
            else:
                yield face.intersection(boundary).area / face.area >= overlap_epsilon

    return points, faces[[*check_faces()]], indices_of_boundary
