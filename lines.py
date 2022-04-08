import numpy as np
from warnings import warn

import scipy.misc
from scipy.interpolate import interp1d
from scipy.spatial.kdtree import KDTree
from scipy.misc import derivative


def arclength_parametrization(line):
    diffs = np.diff(line, axis=0, prepend=line[0:1, :])
    length_elements = np.linalg.norm(diffs, axis=1)
    return np.cumsum(length_elements)


class _Curve:
    def __init__(self, line, parameters):
        self._points = line
        n = line.shape[1]
        self._dims = n
        self._parameters = parameters
        self._arclength = arclength_parametrization(line)
        self._interpolators = [interp1d(self._parameters, self._points[:, i]) for i in range(n)]
        self._kdtree = KDTree(line)

    def __call__(self, t):
        if isinstance(t, (float, int)):
            if t > np.max(self._parameters):
                warn(f"{t} exceeds parameters domain")
            return np.array([
                f(t) for f in self._interpolators
            ])
        else:
            result = np.empty((t.shape[0], self._dims))

            for i, f in enumerate(self._interpolators):
                result[:, i] = f(t)

            return result

    def query(self, t):
        return self(t)

    def distance(self, point):
        _, on_line = self.retract(point)
        return np.linalg.norm(on_line - point)

    def compute_riemannian_length(self, metric, res=None):
        from .riemann import compute_length
        N = self._points.shape[0]
        lens = np.zeros(N - 1)
        for i, j in zip(range(0, N), range(1, N)):
            lens[i] = compute_length(self._points[i, :], self._points[j, :], metric, resolution=res)

        return np.sum(lens)

    def retract(self, point):
        t = self.parameterize(point)
        return t, self(t)

    def derivative(self, t):
        order = 5
        dt = 1e-3
        mi, ma = self.range
        ho = order >> 1
        if t - ho*dt < mi:
            tprime = t + ho*dt
        elif t + ho*dt > ma:
            tprime = t - ho*dt
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
        return self._points.shape

    @property
    def points(self):
        return self._points

    def __iter__(self):
        return iter(self._points)


class ParametricCurve(_Curve):
    def __init__(self, line):
        al = arclength_parametrization(line)
        super().__init__(line, al / al[-1])


class ArcLengthParametrizedCurve(_Curve):
    def __init__(self, line):
        al = arclength_parametrization(line)
        super().__init__(line, al)


class ClosedCurve(_Curve):
    def __init__(self, points):
        if np.linalg.norm(points[0, :] - points[-1, :]) < 1e-12:
            p = points
        else:
            p = np.zeros((points.shape[0] + 1, points.shape[1]))
            p[0:-1, :] = points
            p[-1, :] = points[0, :]
        al = arclength_parametrization(p)
        super().__init__(p, 2 * np.pi * al / al[-1])


class Curve(_Curve):
    def __init__(self, line, params=None):
        if params is None:
            params = arclength_parametrization(line)
        super().__init__(line, params)


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
