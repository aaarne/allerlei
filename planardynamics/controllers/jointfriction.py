import numpy as np


class SimulatedJointFriction:
    def __init__(self, friction_coeff):
        if (type(friction_coeff) is float) or (type(friction_coeff) is int):
            self._k = friction_coeff
        else:
            self._k = None
            self._K = np.diag(friction_coeff)

    def __call__(self, t, q, dq):
        if self._k:
            return -self._k * dq
        else:
            return -self._K @ dq
