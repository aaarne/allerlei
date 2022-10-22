import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.spatial import Delaunay, delaunay_plot_2d

def triangulate_disk_randomly_rejection(r, n, min_dist=1e-3, max_trials=1000):
    points = np.zeros((n, 2))

    def check_up_to_n(p, n):
        return np.all(np.linalg.norm(points[0:n, :]-p, axis=1) > min_dist)

    for i in range(1, n):
        for _ in range(max_trials):
            phi = np.random.uniform(0, 2*pi)
            r = np.sqrt(np.random.uniform(0, 1))
            p = np.array([r*np.cos(phi), r*np.sin(phi)])

            if check_up_to_n(p, i):
                points[i, :] = p
                break
        else:
            raise ArithmeticError("couldn't find new sample")

    return Delaunay(r*points)


def triangulate_disk_randomly(r, n, boundary_ratio=0.05):
    boundary_points = int(np.ceil(boundary_ratio*n))
    phi = np.linspace(0, 2 * pi, boundary_points, endpoint=False)
    boundary = np.empty((boundary_points, 2))
    boundary[:, 0] = np.cos(phi)
    boundary[:, 1] = np.sin(phi)

    interior_points = n - boundary_points - 1

    l = np.sqrt(np.random.uniform(0, 1, size=interior_points))
    phi = np.random.uniform(0, 2 * pi, interior_points)

    interior = np.empty((interior_points, 2))
    interior[:, 0] = l * np.cos(phi)
    interior[:, 1] = l * np.sin(phi)

    points = np.vstack((boundary, interior, np.array([0, 0]).reshape(1, -1)))
    return Delaunay(r*points)


def __sample_circle_circumference(n, r):
    if r == 0:
        return np.array([[0, 0]])
    phi = np.linspace(0, 2 * pi, n, endpoint=False)
    p = np.empty((n, 2))
    p[:, 0] = r * np.cos(phi)
    p[:, 1] = r * np.sin(phi)
    return p


def triangulate_disk_systematic(r, n, return_boundary=False):
    rad_steps = int(np.ceil(np.sqrt(n/pi)))
    tri = Delaunay(np.vstack(__sample_circle_circumference(
        n=int(2*rad*n/rad_steps),
        r=rad,
    ) for rad in np.linspace(0, r, rad_steps)))
    if return_boundary:
        n_outer = int(2*n/rad_steps)
        outer_indices = np.arange(tri.points.shape[0] - n_outer, tri.points.shape[0])
        return tri, outer_indices
    else:
        return tri



def triangulate_disk_with_given_rads(_, n, radii):
    total_length = np.sum(2*pi*radii)
    points_per_unit_length = n / total_length
    return Delaunay(np.vstack(__sample_circle_circumference(
        n=int(np.round(2*pi*r*points_per_unit_length)),
        r=r,
    ) for r in radii)), total_length


def triangulate_disk_planar_cut(r, n):
    n_ax = np.sqrt(4/pi * n)
    s = np.linspace(-1, 1, int(np.ceil(n_ax)))
    xx, yy = np.meshgrid(s, s)
    Q = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
    return Delaunay(r*Q[np.linalg.norm(Q, axis=1) <= 1.0])


def ordered_hull(hull):
    h = hull.copy()
    target, v = h[0, :]
    yield v
    h[0, :] = -1
    while (h >= 0).any():
        v_idx = 0 if (h[:, 0] == v).any() else 1
        v_target_idx = 0 if v_idx else 1
        row_idx = np.argwhere(h[:, v_idx] == v).item()
        v = h[row_idx, v_target_idx]
        h[row_idx, :] = -1
        yield v
    if v != target:
        raise RuntimeError("No cyclic boundary")


if __name__ == "__main__":
    n = 1000

    radii = np.linspace(0, 1, 30)**4

    from functools import partial

    testees = {
        "Pure Random": triangulate_disk_randomly,
        "Rejection Sampling": triangulate_disk_randomly_rejection,
        "Disk Systematic": triangulate_disk_systematic,
        "Planar Cut": triangulate_disk_planar_cut,
        "Given Radii:": partial(triangulate_disk_with_given_rads, radii=radii),
    }

    for title, method in testees.items():
        plt.title(title)
        tri = method(2.0, n)
        print(f"{title}: {tri.points.shape[0]}")
        delaunay_plot_2d(tri, plt.gca())

        plt.figure()

    plt.show()

