import numpy as np
from warnings import warn
from .riemann import compute_length

import scipy.misc
from scipy.interpolate import interp1d
from scipy.spatial.kdtree import KDTree


def arclength_parametrization(line):
    diffs = np.diff(line, axis=0, prepend=line[0:1, :])
    length_elements = np.linalg.norm(diffs, axis=1)
    return np.cumsum(length_elements)


class _Curve:
    def __init__(self, line, parameters, lazy=True):
        self._points = line
        n = line.shape[1]
        self._dims = n
        self._parameters = parameters
        self._arclength = arclength_parametrization(line)
        if lazy:
            self._kdt = None
            self._ints = None
        else:
            self._kdt = KDTree(line)
            self._ints = self._create_interpolators()

    @property
    def _interpolators(self):
        if self._ints is None:
            self._ints = self._create_interpolators()
        return self._ints

    def override_points(self, new_points):
        assert new_points.shape == self._points.shape, f"New shape: {new_points.shape}, old shape: {self._points.shape}"
        self._points = new_points

    def create_new(self, lazy=True):
        return self.__class__(self._points, lazy=lazy)

    @property
    def _kdtree(self):
        if self._kdt is None:
            self._kdt = KDTree(self._points)
        return self._kdt

    def _create_interpolators(self):
        return [interp1d(self._parameters, self._points[:, i]) for i in range(self._dims)]

    def _query_single(self, t):
        return np.array([
            f(t) for f in self._interpolators
        ])

    def _query_vectorized(self, t):
        result = np.empty((t.shape[0], self._dims))

        for i, f in enumerate(self._interpolators):
            result[:, i] = f(t)

        return result

    def __call__(self, t):
        if isinstance(t, (float, int)):
            if t > np.max(self._parameters):
                warn(f"{t} exceeds parameters domain")
            return self._query_single(t)
        else:
            return self._query_vectorized(t)

    def adjacent_points(self, i, j):
        return self._points[i, :], self._points[i, :] + self.difference_vector(i, j)

    def difference_vector(self, i, j):
        return self._points[j, :] - self._points[i, :]

    def unit_difference_vector(self, i, j):
        d = self.difference_vector(i, j)
        return d/np.linalg.norm(d)

    def query(self, t):
        return self(t)

    def distance(self, point):
        _, on_line = self.retract(point)
        return np.linalg.norm(on_line - point)

    def compute_riemannian_length(self, metric=None, res=1e-3):
        N = self._points.shape[0]
        lens = np.zeros(N - 1)
        for i, j in zip(range(0, N), range(1, N)):
            lens[i] = self.riemannian_length_between_indices(metric, i, j, res=res)

        return np.sum(lens)

    def riemannian_length_between_indices(self, metric, i, j, res=1e-3):
        return compute_length(*self.adjacent_points(i, j), metric, resolution=res)

    def retract(self, point):
        t = self.parameterize(point)
        return t, self(t)

    def derivative(self, t):
        order = 5
        dt = 1e-3
        mi, ma = self.range
        ho = order >> 1
        if t - ho * dt < mi:
            tprime = t + ho * dt
        elif t + ho * dt > ma:
            tprime = t - ho * dt
        else:
            tprime = t
        return scipy.misc.derivative(self.__call__, tprime, dx=dt)

    def tangent(self, t):
        d = self.derivative(t)
        return d / np.linalg.norm(d)

    def parameterize(self, point, func=lambda x: x):
        p = func(self._parameters)
        if len(point.shape) == 1:
            d, i = self._kdtree.query(point, k=2)
            w = np.reciprocal(d)
            return (p[i[0]] * w[0] + p[i[1]] * w[1]) / (w[0] + w[1])
        elif len(point.shape) == 2 and point.shape[1] == self._dims:
            d, i = self._kdtree.query(point, k=2)
            w = np.reciprocal(d)
            acc = np.zeros(point.shape[0])
            exact = np.zeros(point.shape[0])
            for k in range(d.shape[1]):
                acc += p[i[:, k]] * w[:, k]
                infs = np.isinf(w[:, k])
                exact[infs] = p[i[infs, k]]
            acc /= np.sum(w, axis=1)

            exact_matches = np.isinf(np.sum(w, axis=1))
            acc[exact_matches] = exact[exact_matches]

            return acc

    @property
    def arclength(self):
        return self._arclength[-1]

    @property
    def parameters(self):
        return self._parameters

    @property
    def range(self):
        return self._parameters[0], self._parameters[-1]

    def resample(self, n):
        sample_points = np.linspace(0, self._parameters[-1], n)
        return self.query(sample_points)

    @property
    def shape(self):
        return self.points.shape

    @property
    def points(self):
        return self._points

    def __getitem__(self, item):
        return self._points[item, :]

    @property
    def n_points(self):
        return self.points.shape[0]

    def __iter__(self):
        return iter(self.points)

    @property
    def dimensions(self):
        return self._dims

    def plot(self, ax, *args, **kwargs):
        ax.plot(self._points[:, 0], self._points[:, 1], *args, **kwargs)


class ParametricCurve(_Curve):
    def __init__(self, line, **kwargs):
        al = arclength_parametrization(line)
        super().__init__(line, al / al[-1], **kwargs)


class ArcLengthParametrizedCurve(_Curve):
    def __init__(self, line, **kwargs):
        al = arclength_parametrization(line)
        super().__init__(line, al, **kwargs)


class _ClosedCurve(_Curve):
    def __init__(self, points, parameters, **kwargs):
        super().__init__(points, parameters, **kwargs)

    def resample(self, n):
        sample_points = np.linspace(0, self._parameters[-1], n, endpoint=False)
        return self.query(sample_points)

    def override_points(self, new_points):
        assert new_points.shape == self.points.shape, f"New shape: {new_points.shape}, old shape: {self.points.shape}"
        self._points[0:-1, :] = new_points
        self._points[-1, :] = new_points[0, :]

    @property
    def points(self):
        return self._points[0:-1, :]


class ClosedCurve(_ClosedCurve):
    def __init__(self, points, **kwargs):
        if np.linalg.norm(points[0, :] - points[-1, :]) < 1e-12:
            p = points
        else:
            p = np.zeros((points.shape[0] + 1, points.shape[1]))
            p[0:-1, :] = points
            p[-1, :] = points[0, :]
        al = arclength_parametrization(p)
        params = 2 * np.pi * al / al[-1]
        super().__init__(points, params, **kwargs)


class Curve(_Curve):
    def __init__(self, line, params=None, **kwargs):
        if params is None:
            params = arclength_parametrization(line)
        super().__init__(line, params, **kwargs)


def discrete_zero_crossing(x, y, target=0.0):
    assert np.shape(x) == np.shape(y)
    assert np.shape(x)[0] >= 3
    crossings, = np.where(np.diff(np.signbit(y - target)))

    def index_to_x_value(i):
        if i < y.size - 2:
            y_sel = y[i:i + 2]
            x_sel = x[i:i + 2]
            direction = 1 if y_sel[0] < y_sel[1] else -1
            x_target = np.interp(target, y_sel[::direction], x_sel[::direction])

            return x_target
        else:
            return x[i]

    return [*map(index_to_x_value, crossings)]
