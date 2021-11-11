import numpy as np
from .joint_configuration_sampler import JointConfigurationSampler


class InverseKinematicSampler(JointConfigurationSampler):
    def __init__(self, robot, x):
        super(InverseKinematicSampler, self).__init__(robot)
        self.x = x

    def set_target(self, x):
        self.x = x

    def stream(self, cartstream):
        for x, q0 in zip(cartstream, super(InverseKinematicSampler, self).forever()):
            try:
                sample = self.robot.bad_invkin(x, q0=q0)
                if self.apply_filters(sample):
                    yield sample
            except AssertionError:
                pass

    def forever(self):
        import itertools
        return self.stream(itertools.repeat(self.x))

    def take_vectorized(self, n, estimated_over_percentage=0.1, verbose=False, max_samples=100000):
        raise NotImplementedError


class InverseKinematicRegionSampler(InverseKinematicSampler):
    def __init__(self, robot, bounds):
        super(InverseKinematicRegionSampler, self).__init__(robot, None)
        self.cartrange = bounds

    def cartesian_range(self, bounds):
        self.cartrange = bounds

    def _sample_cartesian(self):
        return self.cartrange[:, 0] + (self.cartrange[:, 1] - self.cartrange[:, 0]) * np.random.rand(self.cartrange.shape[0])

    def forever(self):
        def cartstream():
            while True:
                yield self._sample_cartesian()

        return self.stream(cartstream())

