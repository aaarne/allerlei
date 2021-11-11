import numpy as np


def Euclidean(x):
    if len(x.shape) == 1:
        return np.eye(2)
    elif len(x.shape) == 2:
        m = np.zeros((x.shape[0], 2, 2))
        m[:, 0, 0] = 1
        m[:, 1, 1] = 1
        return m
    else:
        raise ValueError


def compute_length(start, end, metric, resolution=1e-3):
    """
    Compute length of straight (Euclidean straightness) line segments given a Riemannian metric
    """
    d = end - start
    steps = np.ceil(np.linalg.norm(d) / resolution)
    vector = d / steps
    points = np.array([start + t * d for t in np.linspace(0, 1, int(steps))])
    elements = np.einsum("i,nij,j->n", vector, metric(points), vector)
    elements[(elements < 0) & (elements > -1e-12)] = 0.0
    return np.sum(np.sqrt(elements))
