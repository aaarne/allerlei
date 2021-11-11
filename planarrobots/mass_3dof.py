from numpy import *


def mass_3dof(q, m=array([3, 3, 3]), l=array([1/3, 1/3, 1/3])):
    M = empty((3, 3))
    mx = m * l / 2
    iner = 4 / 12 * m * l ** 2
    xi1 = array([m[0], mx[0], iner[0]])
    xi2 = array([m[1], mx[1], iner[1]])
    xi3 = array([m[2], mx[2], iner[2]])
    t1 = l[0] ** 2
    t3 = cos(q[2])
    t4 = cos(q[1])
    t6 = l[0] * xi3[1]
    t7 = t3 * t4 * t6
    t9 = sin(q[2])
    t10 = sin(q[1])
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
    M[0, 0] = t26
    M[0, 1] = t27
    M[0, 2] = t28
    M[1, 0] = t27
    M[1, 1] = xi3[2] + xi2[2] + t24 + t15
    M[1, 2] = t30
    M[2, 0] = t28
    M[2, 1] = t30
    M[2, 2] = xi3[2]
    return M.T


def mass_2dof(q, m=(1, 1), l=(1, 1)):
    m = mass_3dof(array([0, q[0], q[1]]), m=array([1, m[0], m[1]]), l=array([0, l[0], l[1]]))
    return m[1:, 1:]

