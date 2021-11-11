#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Translation(object):

    @classmethod
    def of_xyz(cls, x, y, z):
        return cls(np.array([x, y, z]))

    def __init__(self, input=None):
        if input is None:
            self.__translation_vector = np.array([0, 0, 0])
        elif type(input) is np.ndarray:
            self.__translation_vector = input.copy()

        assert self.__translation_vector is not None

    @property
    def translation_vector(self):
        return self.__translation_vector.copy()

    @property
    def length(self):
        return np.sqrt(np.sum(self.__translation_vector ** 2))

    def d_transl(self, other):
        t = Translation(self.translation_vector - other.get_translation_vector())
        return t.length

    def __repr__(self):
        return 'Translation: ' + str(self.__translation_vector)

    def __str__(self):
        return self.__repr__()

    def to_ln_string(self):
        return str(self.translation_vector)

    def __add__(self, other):
        return Translation(self.translation_vector + other.get_translation_vector())

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Translation(self.translation_vector - other.get_translation_vector())

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        assert type(other) is float or type(other) is int
        return Translation(self.translation_vector * float(other))

    def __imul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, index):
        return self.translation_vector[index]

    def to_pretty_string(self):

        def f(n):
            return '{:.3f}'.format(n)

        s = ''
        for line in self.translation_vector:
            s += f(line) + '\n'
        return s[:-1]
