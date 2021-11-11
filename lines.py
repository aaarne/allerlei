import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.kdtree import KDTree


def arclength_parametrization(line):
    diffs = np.diff(line, axis=0, prepend=line[0:1, :])
    length_elements = np.linalg.norm(diffs, axis=1)
    return np.cumsum(length_elements)


class Curve:
    def __init__(self, line, mode='arclength'):
        self._points = line
        n = line.shape[1]
        self._dims = n
        self._arclength = arclength_parametrization(line)
        if mode == 'arclength':
            self._parameters = self._arclength
        elif mode == 'unit':
            self._parameters = self._arclength / self._arclength[-1]
        self._interpolators = [interp1d(self._parameters, self._points[:, i]) for i in range(n)]
        self._kdtree = KDTree(line)

    def __call__(self, t):
        if isinstance(t, (float, int)):
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

    def parameterize(self, point, func=lambda x: x):
        p = func(self._parameters)
        if len(point.shape) == 1:
            d, i = self._kdtree.query(point, k=2)
            w = np.reciprocal(d)
            return (p[i[0]]*w[0] + p[i[1]]*w[1]) / (w[0] + w[1])
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


class ParametricCurve(Curve):
    def __init__(self, line):
        super().__init__(line, mode='unit')


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
