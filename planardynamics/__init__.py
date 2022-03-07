from .double_pendulum import DoublePendulum
from .triple_pendulum import TriplePendulum
from .pendulum import StopIntegration
from .misc import sample_phi
from .controllers import *
from .util import plot_pendulum_trajectory

from numpy import pi

settings = {
    "Stretched": {
        "q0": [
            0,
            0,
        ],
        "bounds": [
            [-pi / 2, pi / 2],
            [-2 * pi, 2 * pi],
        ]
    },
    "Elbow": {
        "q0": [
            0,
            pi / 2,
            ],
        "bounds": [
            [-1.4, pi],
            [-2 * pi, 2 * pi],
        ]
    },
    "Shy": {
        "q0": [
            0,
            pi,
        ],
        "bounds": [
            [-pi, pi],
            [-pi, 3 * pi],
        ]
    },
    "Promising": {
        "q0": [
            0,
            3*pi/4,
            ],
        "bounds": [
            [-2, pi],
            [-2 * pi, 2 * pi],
        ]
    },
    "Boomerang": {
        "q0": [
            -pi/6,
            -pi/4
        ],
        "bounds": [
            [-2.5, 0.75],
            [-2*pi, 2*pi]
        ]
    }
}
