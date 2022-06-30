from itertools import *

import numpy as np
from numpy import sin, cos
from functools import reduce
from .. import hom2xyphi
from .planar_jacobians import jacobians
from .mass_3dof import *


class PlanarRobot(object):
    def __init__(self, link_lengths=(1, 1, 1)):
        self._link_lengths = list(link_lengths)

    def __str__(self):
        return f"Planar robot with {len(self._link_lengths)} rotational joints. Link lengths are: {self._link_lengths}."

    @staticmethod
    def __create_trafo(q, l):
        trafo = np.empty((np.size(q), 3, 3))
        trafo[:, 0, 0] = np.cos(q)
        trafo[:, 0, 1] = -np.sin(q)
        trafo[:, 0, 2] = l * np.cos(q)
        trafo[:, 1, 0] = np.sin(q)
        trafo[:, 1, 1] = np.cos(q)
        trafo[:, 1, 2] = l * np.sin(q)
        trafo[:, 2, 0] = 0.0
        trafo[:, 2, 1] = 0.0
        trafo[:, 2, 2] = 1.0
        return trafo

    def link_trafos(self, joint_angles, up_to=None, include_base=False):
        if up_to is None:
            up_to = self.dof
        g = (self.__create_trafo(q, l) for q, l in zip(joint_angles, self._link_lengths))
        base = repeat(np.eye(3), 1 if include_base else 0)
        return chain(base, islice(g, up_to))

    def endeffector_pose(self, joint_angles, up_to=None):
        return reduce(lambda x, y: x @ y, self.link_trafos(joint_angles, up_to=up_to))

    def forward_kinematics(self, joint_angles, up_to=None):
        q = joint_angles.T
        assert len(q) == len(self._link_lengths)
        flange = np.array(self.endeffector_pose(q, up_to=up_to))
        return hom2xyphi(flange)

    def forward_kinematics_for_each_link(self, q):
        a = np.zeros((q.shape[0], self.dof, 3))
        for i in range(self.dof):
            a[:, i, :] = self.forward_kinematics(q, up_to=i+1)

        return a


    def bad_invkin(self, cart, q0=None, K=1.0, tol=1e-3, max_steps=100):
        if q0 is not None:
            q = q0
        else:
            q = np.pi * {
                1: np.array([0]),
                2: np.array([.25, .5]),
                3: np.array([.25, .5, -.5]),
                4: np.array([.25, .5, -.5, .5]),
                5: np.array([.25, .5, -.5, .5, -.5]),
                6: np.array([.25, .5, -.5, .5, -.5, .5])
            }[self.number_of_joints]

        for _ in range(max_steps):
            if cart.size == 3:
                f = lambda x: self.forward_kinematics(x)
                if self.number_of_joints == 3:
                    A = lambda x: np.linalg.inv(self.jacobian(x))
                else:
                    A = lambda x: np.linalg.pinv(self.jacobian(x))
            elif cart.size == 2:
                f = lambda x: self.forward_kinematics(x)[:, 0:2]
                A = lambda x: np.linalg.pinv(self.jacobian(x)[0:2, :])
            elif cart.size == 1:
                f = lambda x: self.forward_kinematics(x)[:, 0:1]
                A = lambda x: np.linalg.pinv(self.jacobian(x)[0:1, :])
            else:
                raise ValueError("Illegal length of desired task-space pose")
            e = cart - f(q)
            if np.linalg.norm(e) < tol:
                break
            try:
                inc = A(q) @ np.squeeze(K * e)
            except ValueError:
                inc = (A(q) * np.squeeze(K * e))[:, 0]
            q += inc
        else:
            raise AssertionError(f"No invkin solution found for {cart}.")

        return np.arctan2(np.sin(q), np.cos(q))

    def smooth_forward_kinematics(self, joint_angles):
        fkin = self.forward_kinematics(joint_angles)
        smooth_fkin = np.zeros((fkin.shape[0] + 1, fkin.shape[1]))
        smooth_fkin[0:2] = fkin[0:2]
        smooth_fkin[2] = np.cos(fkin[2])
        smooth_fkin[3] = np.sin(fkin[2])
        return smooth_fkin

    def jacobian(self, joint_angles):
        raw = jacobians[self.number_of_joints](joint_angles.T, self._link_lengths)
        if len(joint_angles.shape) > 1:
            j = np.zeros((joint_angles.shape[0], 3, self.number_of_joints))
            for i in range(self.number_of_joints):
                j[:, 0, i] = raw[0, i]
                j[:, 1, i] = raw[1, i]
                j[:, 2, i] = raw[2, i]
        else:
            j = raw
        return j

    @property
    def number_of_joints(self):
        return len(self._link_lengths)

    @property
    def dof(self):
        return len(self._link_lengths)
