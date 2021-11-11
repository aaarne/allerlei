import numpy as np


class JointConfigurationSampler:
    def __init__(self, robot):
        self.robot = robot
        self.filters = list()
        i = [[-1, 1] for _ in range(robot.number_of_joints)]
        self._bounds = np.pi * np.array(i)

    def _inf_stream(self):
        while True:
            yield self._bounds[:, 0] + (self._bounds[:, 1] - self._bounds[:, 0]) * np.random.rand(
                self.robot.number_of_joints)

    def apply_filters(self, sample):
        return all(f(sample) for f in self.filters)

    @property
    def bounds(self):
        return self._bounds

    def joint_value_bounds(self, bounds):
        self._bounds = bounds

    def forever(self):
        for sample in self._inf_stream():
            if self.apply_filters(sample):
                yield sample

    def avoid_singularities(self, threshold=0.1, target=None):
        def f(sample):
            J = self.robot.jacobian(sample)
            if len(J.shape) == 2:
                selection = np.arange(J.shape[0]) if target is None else target
                last_singular_value = np.sqrt(np.linalg.svd(J[selection, :], compute_uv=False)[-1])
                return last_singular_value > threshold
            else:
                selection = np.arange(J.shape[1]) if target is None else target
                svds = np.linalg.svd(J[:, selection, :], compute_uv=False)
                # value = np.product(svds, axis=1)
                value = svds[:, -1]
                return value > (threshold ** 2)

        self.filters.append(f)

    def add_filter(self, filt):
        self.filters.append(filt)

    def take(self, n):
        from itertools import islice as take
        return np.array([*take(self.forever(), n)])

    def sample(self):
        return self.take(1).squeeze()

    def take_vectorized(self, n, estimated_over_percentage=0.1, verbose=False, max_samples=100000):

        def loop(n):
            n_sample = min(int(np.ceil(n * (1 + estimated_over_percentage))), max_samples)
            q = self._bounds[:, 0] + (self._bounds[:, 1] - self._bounds[:, 0]) * \
                np.random.rand(n_sample, self.robot.number_of_joints)
            mask = np.ones((n_sample,), dtype=bool)
            for f in self.filters:
                mask = mask & (f(q))

            q = q[mask]

            if verbose:
                print(f"Rejected {n_sample - np.sum(mask)} out of {n_sample} samples.")

            if q.shape[0] > n:
                q = q[0:n]

            return q, q.shape[0]

        n_missing = n
        q_sampled = list()
        while n_missing > 0:
            q_t, n_t = loop(n_missing)
            q_sampled.append(q_t)
            n_missing = n_missing - n_t

        return np.concatenate(q_sampled)
