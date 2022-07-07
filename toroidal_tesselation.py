import numpy as np
from scipy.spatial import Delaunay


def create_toroidal_tesselation(resolution=20, scaling=1, offset=0):
    x = np.arange(resolution + 1)
    y = np.arange(resolution + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).T

    tri = Delaunay(points)

    faces = tri.simplices

    for i, j in zip([0, 1], [1, 0]):
        upper_line, = np.nonzero(points[:, i] == resolution)
        lower_line = np.array(
            [np.nonzero((points[:, i] == 0) & (points[:, j] == points[k, j]))[0].item() for k in upper_line])

        for upper, lower in zip(upper_line, lower_line):
            faces[faces == upper] = lower
    return (points * scaling/resolution) + offset, faces


def parametric_torus(uv, c=3, a=1, compute_jacobian=False):
    points = np.zeros((uv.shape[0], 3))
    u = uv[:, 0]
    v = uv[:, 1]
    alpha = c + a * np.cos(v)

    points[:, 0] = alpha * np.cos(u)
    points[:, 1] = alpha * np.sin(u)
    points[:, 2] = a * np.sin(v)

    if compute_jacobian:
        jac = np.zeros((uv.shape[0], 3, 2))
        jac[:, 0, 0] = -alpha*np.sin(u)
        jac[:, 0, 1] = -a*np.sin(v)*np.cos(u)
        jac[:, 1, 0] = alpha*np.cos(u)
        jac[:, 1, 1] = -a*np.sin(v)*np.sin(u)
        jac[:, 2, 0] = 0
        jac[:, 2, 1] = a*np.cos(v)
        return points, jac
    else:
        return points


def create_2d_angular_torus(resolution, r1, r2):
    p, f = create_toroidal_tesselation(resolution=resolution, scaling=2*np.pi, offset=-np.pi)
    tp = parametric_torus(p, c=r1, a=r2)
    import pymesh
    torus = pymesh.form_mesh(tp, f)
    torus, info = pymesh.remove_isolated_vertices(torus)
    return p[info['ori_vertex_index']], torus