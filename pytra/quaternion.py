# /usr/bin/env python
# -*- coding: utf-8 -*-

from .trafo_utils import *


class Quaternion(object):
    '''Quaternion class to handle Quaternion representation of orientation, operators + - and * are overwritten,
    Constructor: Quaternion(a,b,c,d,normelize=True) unset normalize if Quaternion shall not be an union quaternion.
    Quaternion class contains a factory to create quaterion from rotation_matrix, euler angles or axis-angle representation.'''

    # Factory:
    @staticmethod
    def create_from_axis_angle(K, theta):
        a = np.cos(theta / 2)
        b = np.sin(theta / 2) * K[0, 0]
        c = np.sin(theta / 2) * K[1, 0]
        d = np.sin(theta / 2) * K[2, 0]
        return Quaternion(a, b, c, d)

    @staticmethod
    def of_axis_angle(K, theta):
        return Quaternion.create_from_axis_angle(K, theta)

    @staticmethod
    def create_from_rotation_matrix(rotmat):
        axis, angle = rotation_matrix_to_axis_angle(rotmat)
        return Quaternion.create_from_axis_angle(axis, angle)

    @staticmethod
    def of_rotation_matrix(rotmat):
        return Quaternion.create_from_rotation_matrix(rotmat)

    @staticmethod
    def create_from_euler_angles_rad(euler):
        rotmat = kuka_to_trafo_rad(0, 0, 0, euler[0], euler[1], euler[2])[:3, :3]
        return Quaternion.create_from_rotation_matrix(rotmat)

    @staticmethod
    def of_euler_angles_rad(euler):
        return Quaternion.create_from_euler_angles_rad(euler)

    @staticmethod
    def create_from_euler_angles_deg(euler):
        rotmat = kuka_to_trafo_deg(0, 0, 0, euler[0], euler[1], euler[2])[:3, :3]
        return Quaternion.create_from_rotation_matrix(rotmat)

    @staticmethod
    def of_euler_angles_deg(euler):
        return Quaternion.create_from_euler_angles_deg(euler)

    @staticmethod
    def create_from_quaternion_array(array):
        return Quaternion(array[0], array[1], array[2], array[3])

    @staticmethod
    def of_quaternion_array(qarray):
        return Quaternion.create_from_quaternion_array(qarray)

    @staticmethod
    def create_neutral_quaternion():
        return Quaternion(0.0, 0.0, 0.0, 1.0)

    # EndFactory

    def __init__(self, a, b, c, d, normalize=True):
        self.q1 = a
        self.q2 = b
        self.q3 = c
        self.q4 = d
        if normalize:
            self.normalize()

    def normalize(self):
        length = self.length
        self.q1 /= length
        self.q2 /= length
        self.q3 /= length
        self.q4 /= length

        if self.a < 0:
            self.q1 *= -1.0
            self.q2 *= -1.0
            self.q3 *= -1.0
            self.q4 *= -1.0
        return self

    @property
    def length(self):
        return np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2)

    @property
    def a(self):
        return self.q1

    @property
    def b(self):
        return self.q2

    @property
    def c(self):
        return self.q3

    @property
    def d(self):
        return self.q4

    @property
    def scalar(self):
        return self.a

    @property
    def real(self):
        return self.a

    @property
    def vector(self):
        return np.array([self.b, self.c, self.d])

    @property
    def imag(self):
        return self.vector

    @property
    def angle(self):
        return 2 * np.arccos(self.a)

    @property
    def axis(self):
        if self.angle == 0.0:
            return np.matrix([[1], [0], [0]])
        return 1 / (np.sin(self.angle / 2)) * np.matrix([[self.b], [self.c], [self.d]])

    @property
    def numpy_array(self):
        return np.array([self.a, self.b, self.c, self.d])

    def add(self, quat):
        a = self.a + quat.a
        b = self.b + quat.b
        c = self.c + quat.c
        d = self.d + quat.d
        return Quaternion(a, b, c, d, False)

    def sub(self, quat):
        a = self.a - quat.a
        b = self.b - quat.b
        c = self.c - quat.c
        d = self.d - quat.d
        return Quaternion(a, b, c, d, False)

    def dot(self, quat):
        return self.a * quat.a \
             + self.b * quat.b \
             + self.c * quat.c \
             + self.d * quat.d

    def outer(self, quat):
        a = np.array([self.a, self.b, self.c, self.d])
        b = np.array([quat.a, quat.b, quat.c, quat.d])
        return np.outer(a, b)

    def mul(self, quat, normalize=True):
        a = self.a * quat.a - self.b * quat.b - self.c * quat.c - self.d * quat.d
        b = self.a * quat.b + self.b * quat.a + self.c * quat.d - self.d * quat.c
        c = self.a * quat.c - self.b * quat.d + self.c * quat.a + self.d * quat.b
        d = self.a * quat.d + self.b * quat.c - self.c * quat.b + self.d * quat.a
        return Quaternion(a, b, c, d, normalize)

    def conjugate(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def get_angle_to(self, other):
        a = self.axis
        b = other.axis
        cosa = (a.T.dot(b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        return np.arccos(cosa)

    def get_error_quaternion(self, meanquat):
        return self.conjugate() * meanquat

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        if type(other) is int or type(other) is float:
            return Quaternion(self.a * float(other), self.b * float(other), self.c * float(other),
                              self.d * float(other), False)
        elif type(other) is Quaternion:
            return self.mul(other)
        else:
            raise Exception('unexpected type for quaternion multiplication')

    def __imul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return 'Quaternion: [{0} {1} {2} {3}]'.format(self.a, self.b, self.c, self.d)

    def __repr__(self):
        return self.__str__()
