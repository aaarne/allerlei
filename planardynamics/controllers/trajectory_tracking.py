import numpy as np
from numpy.linalg import inv, pinv, LinAlgError
from .impedance import ImpedanceController
from .utils import *


class TaskFeedForwardController:
    def __init__(self, trajectory, jacobian_fun, mass_fun, cc_fun=None, do_jac_prediction=True):
        self._traj = trajectory
        self._jacobian_fun = jacobian_fun
        self._cc_fun = cc_fun
        self._mass = mass_fun
        self._predict_jac = do_jac_prediction
    
    def __call__(self, t, q, dq):
        J = self._jacobian_fun(q)
        M = self._mass(q)
        M_x = transform_mass(M, J)

        x, v, a = self._traj(t)
        a_ff = a.copy()

        if self._cc_fun or self._predict_jac:
            Minv = inv(M)
            if J.shape[0] == J.shape[1]:
                J_pinv = inv(J)
            else:
                J_pinv = weighted_pinv(J, Minv)

            if self._predict_jac:
                dJ = numerical_grad(lambda t: self._jacobian_fun(q + t*dq), 0, 1e-3)
                a_ff -= dJ @ J_pinv @ v

            if self._cc_fun:
                C = self._cc_fun(q, dq)
                a_ff += J @ Minv @ C @ J_pinv @ v

        if type(M_x) is np.ndarray:
            f_ff = M_x @ a_ff
        else:
            f_ff = M_x * a_ff

        return J.T @ f_ff
        

class TrajectoryTrackingController:
    def __init__(self, trajectory, fkin_fun, jacobian_fun, mass_fun, K, Zeta, cc_fun=None, no_ff=False, **kwargs):
        self._noff = no_ff
        self._feedback = ImpedanceController(
            fkin_fun=fkin_fun,
            jacobian_fun=jacobian_fun,
            mass_fun=mass_fun,
            K=K,
            Zeta=Zeta,
            desired_pos_fun=trajectory.pos,
            desired_vel_fun=trajectory.vel,
        )
        self._feed_forward = TaskFeedForwardController(
            trajectory=trajectory,
            jacobian_fun=jacobian_fun,
            mass_fun=mass_fun,
            cc_fun=cc_fun,
            **kwargs,
        )

    def __call__(self, t, q, dq):
        tau = self._feedback(t, q, dq)

        if not self._noff:
            tau += self._feed_forward(t, q, dq)

        return tau


