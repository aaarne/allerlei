import numpy as np
from numpy import sin, cos

jacobians = {
    1: lambda q, l: np.array([[-l[0] * sin(q[0])], [l[0] * cos(q[0])], [1]]),
    2: lambda q, l: np.array([[-l[0] * sin(q[0]) - l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]),
                               -l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0])],
                              [l[0] * cos(q[0]) - l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]),
                               -l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1])], [1, 1]]),
    3: lambda q, l: np.array([[-l[0] * sin(q[0]) - l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[
        2] * (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]),
                               -l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[2] * (
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]),
                               -l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])], [
                                  l[0] * cos(q[0]) - l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[
                                      2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]),
                                  -l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[2] * (
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]),
                                  l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - l[2] * (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])], [1, 1, 1]]),
    4: lambda q, l: np.array([[-l[0] * sin(q[0]) - l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[
        2] * (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
        q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]),
                               -l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[2] * (
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]),
                               -l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]),
                               l[3] * (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - l[
                                   3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])], [
                                  l[0] * cos(q[0]) - l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[
                                      2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]),
                                  -l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[2] * (
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]),
                                  l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - l[2] * (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]),
                                  -l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) +
                                  l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])],
                              [1, 1, 1, 1]]),
    5: lambda q, l: np.array([[-l[0] * sin(q[0]) - l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[
        2] * (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
        q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[4] * ((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]),
                               -l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[2] * (
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[
                                   4] * ((-(sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                         sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                 q[2])) * cos(q[3])) * sin(q[4]) + l[4] * (((sin(q[0]) * sin(
                                   q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
                                   q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(
                                   q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(
                                   q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]),
                               -l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[
                                   4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                         -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                 q[2])) * sin(q[3])) * cos(q[4]) + l[4] * (((-sin(q[0]) * sin(
                                   q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(
                                   q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(
                                   q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
                                   q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]), l[3] * (
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
            q[3]) - l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[4] * (-(
                    -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]), -l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])], [
                                  l[0] * cos(q[0]) - l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[
                                      2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                      q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4]),
                                  -l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[2] * (
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                      q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4]),
                                  l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - l[2] * (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) + l[4] * ((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                -sin(q[0]) * cos(q[1]) - sin(
                                                                            q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4]), -l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
            q[3]) + l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) + l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (sin(q[0]) * cos(q[1]) + sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(q[4]) + l[4] * (-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]), l[4] * (-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                             q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(
                                                             q[0])) * sin(q[2])) * cos(q[3])) * cos(q[4]) - l[4] * (((
                                                                                                                                 -sin(
                                                                                                                                     q[
                                                                                                                                         0]) * sin(
                                                                                                                             q[
                                                                                                                                 1]) + cos(
                                                                                                                             q[
                                                                                                                                 0]) * cos(
                                                                                                                             q[
                                                                                                                                 1])) * sin(
            q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * sin(q[3])) * sin(q[4])], [1, 1, 1, 1, 1]]),
    6: lambda q, l: np.array([[-l[0] * sin(q[0]) - l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[
        2] * (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
        q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[4] * ((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[5] * (((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + l[
                                   5] * (((-(sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                      (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                          sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                  q[2])) * cos(q[3])) * cos(q[4]) + (-(
                (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((sin(q[0]) * sin(
        q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
        q[3])) * sin(q[4])) * sin(q[5]), -l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[2] * (
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
        q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[4] * ((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[5] * (((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + l[
                                   5] * (((-(sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                      (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                          sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                  q[2])) * cos(q[3])) * cos(q[4]) + (-(
                (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((sin(q[0]) * sin(
        q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
        q[3])) * sin(q[4])) * sin(q[5]), -l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
        q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[4] * ((-(
                -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[4] * (((-sin(
        q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[5] * ((-(
                -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((sin(q[0]) * sin(
        q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
        q[2])) * sin(q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                       (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                   q[2])) * cos(q[3])) * cos(q[4])) * sin(q[5]) + l[5] * (((-(
                -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + (((-sin(
        q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4])) * cos(q[5]),
                               l[3] * (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - l[
                                   3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[
                                   4] * (-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                     -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                         -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                 q[2])) * cos(q[3])) * sin(q[4]) + l[4] * ((-(
                                           -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(
                                   q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(
                                   q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(
                                   q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[5] * ((-(
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                   q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                                     (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) + (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * cos(q[4])) * cos(q[5]) + l[5] * ((-(
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                   q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * cos(q[4]) + (-(
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4])) * sin(q[5]), -l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[5] * ((-(
                    -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) - ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) - ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * sin(q[4])) * sin(q[5]) + l[5] * (-((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]),
                               l[5] * (-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                     (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                         -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                 q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                                           -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                                 -sin(q[0]) * cos(
                                                                                             q[1]) - sin(q[1]) * cos(
                                                                                             q[0])) * cos(q[2])) * cos(
                                   q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * cos(q[4])) * cos(q[5]) - l[5] * (((-(
                                           -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                           -sin(q[0]) * cos(q[1]) - sin(
                                                                                       q[1]) * cos(q[0])) * cos(
                                   q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                   q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * cos(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                                     (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) + (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4])) * sin(q[5])], [
                                  l[0] * cos(q[0]) - l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[
                                      2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                      q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4])) * cos(q[5]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                              -sin(q[0]) * cos(
                                                                                          q[1]) - sin(q[1]) * cos(
                                                                                          q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5]),
                                  -l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[2] * (
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                      q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4])) * cos(q[5]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                              -sin(q[0]) * cos(
                                                                                          q[1]) - sin(q[1]) * cos(
                                                                                          q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5]),
                                  l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - l[2] * (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) + l[4] * ((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                -sin(q[0]) * cos(q[1]) - sin(
                                                                            q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4]) + l[5] * ((-(
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                              -sin(q[0]) * cos(
                                                                                          q[1]) - sin(q[1]) * cos(
                                                                                          q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5]), -l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
            q[3]) + l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) + l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (sin(q[0]) * cos(q[1]) + sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(q[4]) + l[4] * (-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) + l[5] * (((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * sin(q[3])) * sin(q[4]) + (-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * cos(q[3])) * cos(q[4])) * cos(q[5]) + l[5] * (((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (sin(q[0]) * cos(q[1]) + sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + (((-sin(
            q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(
            q[0])) * cos(q[2])) * sin(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4])) * sin(q[5]),
                                  l[4] * (-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) - l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * sin(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(
                                      q[2])) * cos(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4]) + l[5] * (-(-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + l[5] * ((-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5]), -l[5] * ((-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * cos(q[4])) * sin(q[5]) + l[5] * ((-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * sin(q[4])) * cos(q[5])],
                              [1, 1, 1, 1, 1, 1]]),
    7: lambda q, l: np.array([[-l[0] * sin(q[0]) - l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[
        2] * (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
        q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[4] * ((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[5] * (((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + l[
                                   5] * (((-(sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                      (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                          sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                  q[2])) * cos(q[3])) * cos(q[4]) + (-(
                (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((sin(q[0]) * sin(
        q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
        q[3])) * sin(q[4])) * sin(q[5]) + l[6] * ((-((-(sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                 (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(
                                                             q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(
                                                             q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) - (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * sin(q[5]) + (((
                                                                                                                                  -(
                                                                                                                                              sin(
                                                                                                                                                  q[
                                                                                                                                                      0]) * sin(
                                                                                                                                          q[
                                                                                                                                              1]) - cos(
                                                                                                                                          q[
                                                                                                                                              0]) * cos(
                                                                                                                                          q[
                                                                                                                                              1])) * sin(
                                                                                                                              q[
                                                                                                                                  2]) - (
                                                                                                                                              -sin(
                                                                                                                                                  q[
                                                                                                                                                      0]) * cos(
                                                                                                                                          q[
                                                                                                                                              1]) - sin(
                                                                                                                                          q[
                                                                                                                                              1]) * cos(
                                                                                                                                          q[
                                                                                                                                              0])) * cos(
                                                                                                                              q[
                                                                                                                                  2])) * sin(
        q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(q[4]) + (-(
                (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((sin(q[0]) * sin(
        q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
        q[3])) * sin(q[4])) * cos(q[5])) * sin(q[6]) + l[6] * ((((-(
                sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
        q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + (((sin(
        q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(
        q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + (((
                                                                                                                                  -(
                                                                                                                                              sin(
                                                                                                                                                  q[
                                                                                                                                                      0]) * sin(
                                                                                                                                          q[
                                                                                                                                              1]) - cos(
                                                                                                                                          q[
                                                                                                                                              0]) * cos(
                                                                                                                                          q[
                                                                                                                                              1])) * sin(
                                                                                                                              q[
                                                                                                                                  2]) - (
                                                                                                                                              -sin(
                                                                                                                                                  q[
                                                                                                                                                      0]) * cos(
                                                                                                                                          q[
                                                                                                                                              1]) - sin(
                                                                                                                                          q[
                                                                                                                                              1]) * cos(
                                                                                                                                          q[
                                                                                                                                              0])) * cos(
                                                                                                                              q[
                                                                                                                                  2])) * sin(
        q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(q[4]) + (-(
                (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((sin(q[0]) * sin(
        q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
        q[3])) * sin(q[4])) * sin(q[5])) * cos(q[6]),
                               -l[1] * sin(q[0]) * cos(q[1]) - l[1] * sin(q[1]) * cos(q[0]) + l[2] * (
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[
                                   4] * ((-(sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                         sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                 q[2])) * cos(q[3])) * sin(q[4]) + l[4] * (((sin(q[0]) * sin(
                                   q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
                                   q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(
                                   q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(
                                   q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[5] * (((-(
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(
                                   q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((sin(
                                   q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(
                                   q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + (((sin(
                                   q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(
                                   q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(
                                   q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(
                                   q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + l[5] * (((-(
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(
                                   q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((sin(
                                   q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(
                                   q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(q[4]) + (-(
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4])) * sin(q[5]) + l[6] * ((-((-(
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                                                                                             -sin(q[0]) * cos(
                                                                                         q[1]) - sin(q[1]) * cos(
                                                                                         q[0])) * cos(q[2])) * sin(
                                   q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * sin(q[4]) - (((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) + (
                                                                                 sin(q[0]) * cos(q[1]) + sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * cos(q[4])) * sin(q[5]) + (((-(
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                                                                                    -sin(q[0]) * cos(q[1]) - sin(
                                                                                q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                   q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * cos(q[4]) + (-(
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4])) * cos(q[5])) * sin(q[6]) + l[6] * ((((-(
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(
                                   q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((sin(
                                   q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(
                                   q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + (((sin(
                                   q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(
                                   q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(
                                   q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(q[1]) + sin(
                                   q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + (((-(
                                           sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(
                                   q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((sin(
                                   q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (sin(q[0]) * cos(
                                   q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(q[4]) + (-(
                                           (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                           sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4])) * sin(q[5])) * cos(q[6]),
                               -l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + l[2] * (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2]) + l[3] * (
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[
                                   4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                         -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                 q[2])) * sin(q[3])) * cos(q[4]) + l[4] * (((-sin(q[0]) * sin(
                                   q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(
                                   q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(
                                   q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(
                                   q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[5] * ((-(
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * cos(q[4])) * sin(q[5]) + l[5] * (((-(
                                           -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                           -sin(q[0]) * cos(q[1]) - sin(
                                                                                       q[1]) * cos(q[0])) * cos(
                                   q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * cos(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * sin(q[4])) * cos(q[5]) + l[6] * (((-(
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * cos(q[4])) * sin(q[5]) + (((-(
                                           -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                    -sin(q[0]) * cos(q[1]) - sin(
                                                                                q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * cos(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * sin(q[4])) * cos(q[5])) * cos(q[6]) + l[6] * (((-(
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * cos(q[4])) * cos(q[5]) + (-((-(
                                           -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                     -sin(q[0]) * cos(q[1]) - sin(
                                                                                 q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                   q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                   q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                   q[2]) - (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                                     (sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(
                                                                         q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                   q[3])) * sin(q[4])) * sin(q[5])) * sin(q[6]), l[3] * (
                                           -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                               -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
            q[3]) - l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + l[4] * (-(
                    -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + l[5] * ((-(
                    -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(
            q[5]) + l[5] * ((-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                         -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                             -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
            q[3])) * cos(q[4]) + (-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
            q[3])) * sin(q[4])) * sin(q[5]) + l[6] * ((-(-(
                    -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) - ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * sin(
            q[5]) + ((-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                  -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                      -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(
            q[4]) + (-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                     -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(
            q[4])) * cos(q[5])) * sin(q[6]) + l[6] * (((-(
                    -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(
            q[5]) + ((-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                  -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                      -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(
            q[4]) + (-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                     -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(
            q[4])) * sin(q[5])) * cos(q[6]), -l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * sin(q[4]) + l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
            q[2])) * sin(q[3])) * cos(q[4]) + l[5] * ((-(
                    -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) - ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) - ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * sin(q[4])) * sin(q[5]) + l[5] * (-((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(
            q[5]) + l[6] * (((-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) - (
                                          (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
            q[3])) * cos(q[4]) - ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
            q[3])) * sin(q[4])) * sin(q[5]) + (-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                             (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                         q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                         q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(
            q[5])) * cos(q[6]) + l[6] * (((-(-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) - (
                                                       (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                           -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                   q[2])) * cos(q[3])) * cos(q[4]) - ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(q[4])) * cos(
            q[5]) + (((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                  (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                      -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(
            q[4]) - ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                     -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(
            q[4])) * sin(q[5])) * sin(q[6]), l[5] * (-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) - l[5] * (((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(q[4])) * sin(
            q[5]) + l[6] * (-(-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                            (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
            q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
            q[3])) * cos(q[4])) * sin(q[5]) + (-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                             (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                         q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                         q[2])) * cos(q[3])) * cos(q[4]) - ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(q[4])) * cos(
            q[5])) * sin(q[6]) + l[6] * ((-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                        (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                            -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                    q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(
            q[5]) - (((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                  (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                      -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(
            q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                     -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(
            q[4])) * sin(q[5])) * cos(q[6]), -l[6] * ((-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * cos(q[4])) * sin(q[5]) + (((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(q[4])) * cos(
            q[5])) * sin(q[6]) + l[6] * ((-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                        (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                            -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                                    q[2])) * cos(q[3])) * sin(q[4]) + ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(
            q[5]) - (((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                  (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                      -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(
            q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                     -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(
            q[4])) * sin(q[5])) * cos(q[6])], [
                                  l[0] * cos(q[0]) - l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[
                                      2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                      q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4])) * cos(q[5]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                              -sin(q[0]) * cos(
                                                                                          q[1]) - sin(q[1]) * cos(
                                                                                          q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5]) + l[6] * ((-((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                                -sin(q[0]) * cos(
                                                                                            q[1]) - sin(q[1]) * cos(
                                                                                            q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                       -sin(q[0]) * cos(q[1]) - sin(
                                                                                   q[1]) * cos(q[0])) * sin(
                                      q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5])) * sin(q[6]) + l[6] * ((((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(
                                      q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + ((sin(
                                      q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(
                                      q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3])) * sin(q[4]) + (((-sin(
                                      q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(
                                      q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(
                                      q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
                                      q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(
                                      q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + ((sin(
                                      q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(
                                      q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5])) * cos(q[6]),
                                  -l[1] * sin(q[0]) * sin(q[1]) + l[1] * cos(q[0]) * cos(q[1]) + l[2] * (
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + l[2] * (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + l[3] * ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * cos(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(
                                      q[2])) * cos(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                 -sin(q[0]) * cos(q[1]) - sin(
                                                                             q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4])) * cos(q[5]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                              -sin(q[0]) * cos(
                                                                                          q[1]) - sin(q[1]) * cos(
                                                                                          q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5]) + l[6] * ((-((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                                -sin(q[0]) * cos(
                                                                                            q[1]) - sin(q[1]) * cos(
                                                                                            q[0])) * sin(q[2])) * sin(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * sin(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                                                                       -sin(q[0]) * cos(q[1]) - sin(
                                                                                   q[1]) * cos(q[0])) * sin(
                                      q[2])) * sin(q[3]) + ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5])) * sin(q[6]) + l[6] * ((((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(
                                      q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + ((sin(
                                      q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(
                                      q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3])) * sin(q[4]) + (((-sin(
                                      q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (-sin(q[0]) * cos(
                                      q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) + ((sin(q[0]) * sin(
                                      q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(q[1]) - sin(
                                      q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) + (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (-sin(
                                      q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3]) + ((sin(
                                      q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (-sin(q[0]) * cos(
                                      q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3])) * cos(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3]) - ((sin(q[0]) * sin(q[1]) - cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5])) * cos(q[6]),
                                  l[2] * (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - l[2] * (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2]) + l[3] * (
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) +
                                  l[4] * ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) + l[4] * ((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                -sin(q[0]) * cos(q[1]) - sin(
                                                                            q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4]) + l[5] * ((-(
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + l[5] * (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                              -sin(q[0]) * cos(
                                                                                          q[1]) - sin(q[1]) * cos(
                                                                                          q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5]) + l[6] * (((-(
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + (((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                       -sin(q[0]) * cos(q[1]) - sin(
                                                                                   q[1]) * cos(q[0])) * cos(
                                      q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5])) * cos(q[6]) + l[6] * (((-(
                                              -(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  -sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * cos(q[5]) + (-((-(
                                              -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                                                        -sin(q[0]) * cos(q[1]) - sin(
                                                                                    q[1]) * cos(q[0])) * cos(
                                      q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) - ((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (-sin(q[0]) * cos(q[1]) - sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + (-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5])) * sin(q[6]), -l[3] * (
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
            q[3]) + l[3] * ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3]) + l[4] * ((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (sin(q[0]) * cos(q[1]) + sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(q[4]) + l[4] * (-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) + l[5] * (((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * sin(q[3])) * sin(q[4]) + (-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * cos(q[3])) * cos(q[4])) * cos(q[5]) + l[5] * (((-(
                    -sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (sin(q[0]) * cos(q[1]) + sin(
            q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4]) + (((-sin(
            q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(
            q[0])) * cos(q[2])) * sin(q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * sin(q[4])) * sin(q[5]) +
                                  l[6] * ((-((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                         (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                     q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                     q[2])) * sin(q[3])) * sin(q[4]) - (-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4])) * sin(q[5]) + (((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * sin(q[3])) * cos(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) - (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * cos(q[3])) * sin(q[4])) * cos(q[5])) * sin(q[6]) + l[
                                      6] * ((((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - (
                                                          (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                      q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                      q[2])) * sin(q[3])) * sin(q[4]) + (-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4])) * cos(q[5]) + (((-(-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
            q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) - ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * sin(q[3])) * cos(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) - (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * cos(q[3])) * sin(q[4])) * sin(q[5])) * cos(q[6]),
                                  l[4] * (-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) - l[4] * (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(
                                      q[1])) * sin(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(
                                      q[2])) * cos(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                      q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4]) + l[5] * (-(-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + l[5] * ((-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5]) + l[6] * ((-(-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * sin(q[5]) + ((-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * cos(q[5])) * cos(q[6]) + l[6] * ((-(-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * sin(q[4]) + (-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) - ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * cos(q[4])) * cos(q[5]) + (-(-(
                                              (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                                                  sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(
                                      q[3])) * cos(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(
                                      q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
                                      q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                              sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(
                                      q[3])) * sin(q[4])) * sin(q[5])) * sin(q[6]), -l[5] * ((-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * cos(q[4])) * sin(q[5]) + l[5] * ((-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * sin(q[4])) * cos(q[5]) + l[6] * ((-(-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * sin(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * cos(q[4])) * cos(q[5]) - ((-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * sin(q[4])) * sin(q[5])) * sin(q[6]) + l[
                                      6] * (-((-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * cos(q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(
            q[2])) * cos(q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * sin(
            q[5]) + ((-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                  (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                      sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(
            q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                     sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(
            q[4])) * cos(q[5])) * cos(q[6]), l[6] * (-((-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * cos(q[4])) * sin(q[5]) + ((-(
                    (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                        sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + ((-sin(q[0]) * sin(
            q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
            q[2])) * cos(q[3])) * cos(q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                                           (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                       q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                       q[2])) * sin(q[3])) * sin(q[4])) * cos(q[5])) * cos(q[6]) - l[
                                      6] * (((-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                                          (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(
                                                      q[2]) - (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(
                                                      q[2])) * cos(q[3])) * sin(q[4]) + (((-sin(q[0]) * sin(q[1]) + cos(
            q[0]) * cos(q[1])) * sin(q[2]) + (sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(
            q[3]) + ((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * cos(q[4])) * cos(
            q[5]) + ((-((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * sin(q[3]) + (
                                  (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                      sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * cos(q[3])) * cos(
            q[4]) - (((-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * sin(q[2]) + (
                    sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * cos(q[2])) * cos(q[3]) + (
                                 (-sin(q[0]) * sin(q[1]) + cos(q[0]) * cos(q[1])) * cos(q[2]) - (
                                     sin(q[0]) * cos(q[1]) + sin(q[1]) * cos(q[0])) * sin(q[2])) * sin(q[3])) * sin(
            q[4])) * sin(q[5])) * sin(q[6])], [1, 1, 1, 1, 1, 1, 1]]),
}
