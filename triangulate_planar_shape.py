import numpy as np
from .triangulate_unit_disk import triangulate_disk_systematic
from . import ParametricCurve


def triangulate_planar_shape(boundary, n):
    assert np.linalg.norm(boundary[0, :] - boundary[:, 1]) < 1e-3, "Start and end point should be the same."
    bc = ParametricCurve(boundary)

    tri, bound = triangulate_disk_systematic(n, return_boundary=True)

