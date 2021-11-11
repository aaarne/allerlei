import numpy as np


class Plane:
    def __init__(self, support, normal):
        self._n = normal
        n0 = normal / np.linalg.norm(normal)
        self._n0 = -n0 if np.dot(normal, support) < 0 else n0
        self._d0 = np.dot(support, self._n0)
        self._d = np.dot(support, self._n)

    def distance_to(self, p):
        return np.dot(self._n0, p) - self._d0

    @property
    def unit_normal(self):
        return self._n0

    @property
    def normal(self):
        return self._n

    @property
    def distance_from_origin(self):
        return self._d0

    @property
    def plane_equation(self):
        return np.array([*self._n, self._d])

    @property
    def hesse_normal_form(self):
        return np.array([*self._n0, self._d0])

    @classmethod
    def of_parametric_form(cls, support, w1, w2):
        return cls(support, np.cross(w1, w2))

    def __str__(self):
        return f"Plane with normal {self.normal} and distance {self.distance_from_origin} from origin."
