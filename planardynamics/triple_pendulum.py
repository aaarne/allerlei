import numpy as np
from .pendulum import Pendulum


class TriplePendulum(Pendulum):

    @classmethod
    def create_default(cls):
        m = np.array([3, 3, 3])
        l = np.array([1 / 3, 1 / 3, 1 / 3])

        return cls(
            lengths=l,
            masses=m,
            mx=m * l / 2,
            iner=4 / 12 * m * l ** 2,
            eg=np.array([0, -1, 0]),
            g0=9.81,
        )

    def __init__(self, lengths, masses, mx, iner, eg, g0):
        l, m = lengths, masses
        xi1 = np.array([m[0], mx[0], iner[0]])
        xi2 = np.array([m[1], mx[1], iner[1]])
        xi3 = np.array([m[2], mx[2], iner[2]])
        params = dict(
            lenghts=lengths,
            masses=masses,
            mx=mx,
            iner=iner,
            eg=eg,
            g0=g0,
        )

        def mass(q):
            t1 = l[0] ** 2
            t3 = np.cos(q[2])
            t4 = np.cos(q[1])
            t6 = l[0] * xi3[1]
            t7 = t3 * t4 * t6
            t9 = np.sin(q[2])
            t10 = np.sin(q[1])
            t12 = t9 * t10 * t6
            t14 = l[1] ** 2
            t15 = t14 * xi3[0]
            t16 = t4 * l[0]
            t18 = t16 * l[1] * xi3[0]
            t20 = t16 * xi2[1]
            t23 = t3 * l[1] * xi3[1]
            t24 = 0.2e1 * t23
            t26 = xi3[2] + t1 * xi2[0] + 0.2e1 * t7 - 0.2e1 * t12 + xi2[2] + xi1[
                2] + t15 + 0.2e1 * t18 + 0.2e1 * t20 + t24 + t1 * xi3[0]
            t27 = xi2[2] + t20 + t15 + t18 - t12 + t24 + xi3[2] + t7
            t28 = xi3[2] - t12 + t7 + t23
            t30 = xi3[2] + t23
            return np.array([
                [t26, t27, t28],
                [t27, xi3[2] + xi2[2] + t24 + t15, t30],
                [t28, t30, xi3[2]],
            ])

        def cc(q, q_dot):
            qp = q_dot
            t1 = np.cos(q[2])
            t2 = np.sin(q[1])
            t3 = t1 * t2
            t5 = qp[1] * l[0] * xi3[1]
            t6 = t3 * t5
            t7 = np.sin(q[2])
            t8 = np.cos(q[1])
            t9 = t7 * t8
            t11 = qp[2] * l[0] * xi3[1]
            t12 = t9 * t11
            t13 = t3 * t11
            t14 = t2 * qp[1]
            t16 = l[0] * l[1] * xi3[0]
            t17 = t14 * t16
            t19 = l[1] * xi3[1]
            t20 = t7 * qp[2] * t19
            t21 = t9 * t5
            t22 = l[0] * xi2[1]
            t23 = t14 * t22
            t25 = qp[0] * l[0]
            t26 = t25 * xi3[1]
            t27 = t9 * t26
            t28 = t2 * qp[0]
            t29 = t28 * t22
            t30 = t28 * t16
            t31 = t3 * t26
            t32 = -t13 - t27 - t29 - t6 - t12 - t21 - t30 - t17 - t31 - t23 - t20
            t33 = qp[0] + qp[1] + qp[2]
            c1 = -t6 - t12 - t13 - t17 - t20 - t21 - t23
            c2 = t32
            c3 = -t33 * (t3 * l[0] + t7 * l[1] + t9 * l[0]) * xi3[1]
            c4 = t31 + t29 - t20 + t30 + t27
            c5 = -t20
            c6 = -t7 * t33 * t19
            c7 = (t9 * t25 + t3 * t25 + t7 * qp[1] * l[1] + t7 * qp[0] * l[1]) * xi3[1]
            c8 = t7 * (qp[0] + qp[1]) * t19
            c9 = 0.0e0

            return np.array([
                [c1, c2, c3],
                [c4, c5, c6],
                [c7, c8, c9],
            ])

        def gravity(q):
            t1 = np.sin(q[0])
            t2 = eg[0] * t1
            t4 = np.cos(q[0])
            t5 = eg[0] * t4
            t6 = np.sin(q[1])
            t7 = t6 * xi2[1]
            t8 = t5 * t7
            t9 = np.cos(q[1])
            t10 = t9 * xi2[1]
            t11 = t2 * t10
            t12 = eg[1] * t1
            t13 = t12 * t7
            t14 = eg[1] * t4
            t15 = t14 * t10
            t16 = l[0] * xi2[0]
            t19 = l[0] * xi3[0]
            t22 = np.sin(q[2])
            t23 = eg[0] * t22
            t24 = t1 * t6
            t25 = t24 * xi3[1]
            t26 = t23 * t25
            t27 = np.cos(q[2])
            t28 = eg[0] * t27
            t29 = t4 * t6
            t30 = t29 * xi3[1]
            t31 = t28 * t30
            t32 = t2 * xi1[1] + t8 + t11 + t13 - t15 - t14 * t16 + t2 * t16 - t14 * t19 + t2 * t19 - t26 + t31
            t33 = t1 * t9
            t34 = t33 * xi3[1]
            t35 = t28 * t34
            t37 = t9 * l[1] * xi3[0]
            t38 = t2 * t37
            t39 = eg[1] * t22
            t40 = t39 * t30
            t42 = t6 * l[1] * xi3[0]
            t43 = t12 * t42
            t44 = eg[1] * t27
            t45 = t4 * t9
            t46 = t45 * xi3[1]
            t47 = t44 * t46
            t48 = t39 * t34
            t49 = t5 * t42
            t50 = t44 * t25
            t51 = t14 * t37
            t53 = t23 * t46
            t54 = t35 + t38 + t40 + t43 - t47 + t48 + t49 + t50 - t51 - t14 * xi1[1] + t53
            t57 = t53 - t26 + t50 + t8 + t13 + t35 + t49 - t15 + t40 + t48 + t11 - t51 - t47 + t43 + t38 + t31

            return g0 * np.array([
                t32 + t54,
                t57,
                (t28 * t29 + t28 * t33 + t23 * t45 - t23 * t24 - t44 * t45 + t44 * t24 + t39 * t29 + t39 * t33) * xi3[
                    1],
            ])

        super().__init__(
            lengths=l,
            params=params,
            mass_fun=mass,
            cc_fun=cc,
            gravity_fun=gravity
        )
