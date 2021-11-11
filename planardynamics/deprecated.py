import numpy as np


def create_myphys_double_pendulum(m1=1, m2=1, l1=1, l2=1, g=0, **kwargs):
    def pendulum_ode(_, x):
        sin, cos = np.sin, np.cos
        theta1, omega1, theta2, omega2 = x[0], x[1], x[2], x[3]

        t1 = -g * (2 * m1 + m2) * sin(theta1)
        t2 = -m2 * g * sin(theta1 - 2 * theta2)
        t3 = -2 * sin(theta1 - theta2) * m2 * (omega2 ** 2 * l2 + omega1 ** 2 * l1 * cos(theta1 - theta2))
        t4 = 2 * m1 + m2 - m2 * cos(2 * theta1 - 2 * theta2)
        alpha1 = (t1 + t2 + t3) / (l1 * t4)

        t5 = 2 * sin(theta1 - theta2)
        t6 = omega1 ** 2 * l1 * (m1 + m2)
        t7 = g * (m1 + m2) * cos(theta1)
        t8 = omega2 ** 2 * l2 * m2 * cos(theta1 - theta2)
        alpha2 = (t5 * (t6 + t7 + t8)) / (l2 * t4)

        return np.array([
            omega1,
            alpha1,
            omega2,
            alpha2,
        ])

    return pendulum_ode
