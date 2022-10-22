#!/usr/bin/env python

import numpy as np
from scipy import linalg as la


def R_z(alpha):
    '''Calculate rotation matrix of rotation about Z by given angle'''
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)
    return np.matrix([[calpha, -salpha, 0, 0], [salpha, calpha, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def R_y(beta):
    '''Calculate rotation matrix of rotation about Y by given angle'''
    cbeta = np.cos(beta)
    sbeta = np.sin(beta)
    return np.matrix([[cbeta, 0, sbeta, 0], [0, 1, 0, 0], [-sbeta, 0, cbeta, 0], [0, 0, 0, 1]])


def R_x(gamma):
    '''Calculate rotation matrix of rotation about X by given angle'''
    cgamma = np.cos(gamma)
    sgamma = np.sin(gamma)
    return np.matrix([[1, 0, 0, 0], [0, cgamma, -sgamma, 0], [0, sgamma, cgamma, 0], [0, 0, 0, 1]])


def T_t(x, y, z):
    '''return homogeneous transform containing given translation'''
    return np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def kuka_to_trafo_deg(x, y, z, alpha, beta, gamma):
    '''calculate homogeneous transform of translation and rotation given in kuka convention,
    alpha,beta,gamma are interpreted as degrees'''
    alpharad = np.deg2rad(alpha)
    betarad = np.deg2rad(beta)
    gammarad = np.deg2rad(gamma)
    return kuka_to_trafo_rad(x, y, z, alpharad, betarad, gammarad)


def kuka_to_trafo_rad(x, y, z, alpha, beta, gamma):
    '''calculate homogeneous transform of translation and rotation given in kuka convention,
    alpha,beta,gamma are interpreted as radians'''
    return T_t(x, y, z) * R_z(alpha) * R_y(beta) * R_x(gamma)


def trafo_to_kuka_rad(trafo):
    '''calculate transformation given as homogeneous transform into kuka convention,
    angles are returned as radians'''
    x = trafo[0, 3]
    y = trafo[1, 3]
    z = trafo[2, 3]
    beta = np.arctan2(-trafo[2, 0], np.sqrt(trafo[0, 0] ** 2 + trafo[1, 0] ** 2))
    alpha = np.arctan2(trafo[1, 0] / np.cos(beta), trafo[0, 0] / np.cos(beta))
    gamma = np.arctan2(trafo[2, 1] / np.cos(beta), trafo[2, 2] / np.cos(beta))
    return np.array([x, y, z, alpha, beta, gamma])


def trafo_to_kuka_deg(trafo):
    '''calculate transformation given as homogeneous transform into kuka convention,
    angles are returned as degrees'''
    result = trafo_to_kuka_rad(trafo)
    result[3:] = np.rad2deg(result[3:])
    return result


def rotation_matrix_to_trafo(mat):
    mat = mat.flatten()
    trafo = [[mat[0, i], mat[0, i + 1], mat[0, i + 2], 0] for i in [0, 3, 6]]
    trafo.append([0, 0, 0, 1])
    return np.matrix(trafo)


def rotation_matrix_to_euler_angles_rad(mat):
    res = trafo_to_kuka_rad(rotation_matrix_to_trafo(mat))
    return res[3:6]


def rotation_matrix_to_euler_angles_deg(mat):
    return np.rad2deg(rotation_matrix_to_euler_angles_rad(mat))


def joint_to_deg(joints):
    '''translate given radians joint vector to degrees joint vector'''
    return np.rad2deg(joints)


def trafo_to_axis_angle(trafo):
    '''calculate axis-angle representation of rotation given by homogeneous transform
    returns: tuple of (axis,angle)'''
    return rotation_matrix_to_axis_angle(trafo[:3, :3])


def rotation_matrix_to_axis_angle(mat):
    '''calculate axis-angle representation of rotation given by rotation matrix,
    returns: tuple of (axis,angle)'''
    assert mat.shape == (3, 3), 'matrix has to be 3x3'
    theta = np.arccos((mat[0, 0] + mat[1, 1] + mat[2, 2] - 1.0) / 2.0)
    K = np.matrix([[mat[2, 1] - mat[1, 2]], [mat[0, 2] - mat[2, 0]], [mat[1, 0] - mat[0, 1]]])
    if theta == 0:
        K_n = np.matrix([[1], [0], [0]])
    else:
        K_n = (1 / (2 * np.sin(theta))) * K
        K_n = (1 / (np.sqrt(K_n[0, 0] ** 2 + K_n[1, 0] ** 2 + K_n[2, 0] ** 2))) * K_n
    return K_n, theta


def axis_angle_to_rotation_matrix(axis, angle):
    sk = np.matrix([[0, -axis[2, 0], axis[1, 0]], [axis[2, 0], 0, -axis[0, 0]], [-axis[1, 0], axis[0, 0], 0]])
    sk = sk * float(angle)
    return la.expm(sk)
