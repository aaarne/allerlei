import ray
import numpy as np
from .pendulum import StopIntegration
from ..tools import Timer
from .misc import sample_phi
from skimage.measure import find_contours


@ray.remote
def compute_geodesic(phi, pendulum, q0, t_end, dt, bounds):
    def cb(t, q, qdot):
        if q[0] < bounds[0][0] \
                or q[0] > bounds[0][1] \
                or q[1] < bounds[1][0] \
                or q[1] > bounds[1][1]:
            raise StopIntegration

    dq0 = np.array([np.cos(phi), np.sin(phi)])
    _, traj, _ = pendulum.sim(q0, dq0,
                              callback=cb,
                              t_max=t_end,
                              dt=dt,
                              dt_sigma=1e-2,
                              )
    return traj


@ray.remote
def find_closest(q, qs):
    dists = np.linalg.norm(qs - q, axis=1)
    return np.argmin(dists)


class GeodesicGenerator:
    def __init__(self, pendulum, bounds=None, t_max=1000, dt=0.02, q0=None):
        assert q0 is not None
        self.bounds = np.array([
            [-np.pi / 2, np.pi / 2],
            [-np.pi, np.pi],
        ]) if bounds is None else bounds

        self._tmax = t_max
        self.q0 = q0.squeeze()
        self.dt = dt
        self.pendulum = pendulum
        self._pendulum = ray.put(pendulum)
        self._bounds = ray.put(bounds)

    def compute_geodesic(self, phi):
        return ray.get(
            compute_geodesic.remote(phi, self._pendulum, self.q0, self._tmax, self.dt, self._bounds)
        )

    def labelled_geodesics(self, phi):
        for φ, traj in zip(phi, self.compute_geodesics(phi)):
            a, b = np.cos(2 * φ), np.sin(2 * φ)
            q = traj[:, 0::2]
            # dists = np.linalg.norm(q - self.q0, axis=1)
            dists = np.sum((q - self.q0) ** 2, axis=1)
            labels = np.vstack((a * dists, b * dists)).T
            yield labels, traj

    def compute_geodesics(self, phis):
        return ray.get([
            compute_geodesic.remote(φ, self._pendulum, self.q0, self._tmax, self.dt, self._bounds) for φ in phis
        ])


class UniformTrajectorySampler:
    def __init__(self, geodesic_generator, db_size=1000):
        self._geogen = geodesic_generator
        self._db_size = db_size
        self._db = None
        self._db_labels = None

    def generate_samples(self, n, compute_labels=False, new_db=True):
        if new_db or self._db is None:
            phi = sample_phi(self._db_size)

            trajs, labels = list(), list()
            for lab, traj in self._geogen.labelled_geodesics(phi):
                trajs.append(traj)
                labels.append(lab)
            self._db = np.vstack(trajs)
            self._db_labels = np.vstack(labels)

        qs = ray.put(self._db[:, 0::2])

        geos = np.empty((n, 4))
        if compute_labels:
            labels = np.empty((n, 2))
        bounds = self._geogen.bounds
        x = np.random.uniform(size=(n, 2)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        for i, idx in enumerate(ray.get([find_closest.remote(q, qs) for q in x])):
            geos[i, :] = self._db[idx, :]
            if compute_labels:
                labels[i, :] = self._db_labels[idx, :]

        if compute_labels:
            return geos, labels
        else:
            return geos

    def take(self, n):
        self._geos = self.generate_samples(n)
        return self._geos[:, 0::2]

    def velocities(self, _):
        Rot90 = np.array([[0, -1], [1, 0]])
        return (Rot90 @ self._geos[:, 1::2].T).T.reshape((-1, 1, 2))

    def inv_masses(self, _):
        q = self._geos[:, 0::2]
        m = np.empty((q.shape[0], 2, 2))
        for i in range(q.shape[0]):
            m[i, :, :] = self._geogen.pendulum.M(q[i, :])
        return np.linalg.inv(m)


class ParabolaPretrainer:
    def __init__(self, q0, bounds):
        self.bounds = bounds
        self.q0 = q0

    steps = 10000
    resample_every_nth = 1

    def sample(self, i=0, n_samples=1000):
        bounds = self.bounds
        x = np.random.uniform(size=(n_samples, 2)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        # target = -multivariate_normal.pdf(x, mean=np.array([0, 0])).reshape((-1, 1))*np.sqrt(2*np.pi)
        zero_pos = self.q0
        target = ((x[:, 0] - zero_pos[0]) ** 2 + (x[:, 1] - zero_pos[1]) ** 2).reshape((-1, 1))
        return x, target


def eqipotential_start_positions(potential, bounds, n, dU, q0, res):
    U0 = potential.predict(q0)
    q1 = np.linspace(bounds[0][0], bounds[0][1], res)
    q2 = np.linspace(bounds[1][0], bounds[1][1], res)
    Q1, Q2 = np.meshgrid(q1, q2)
    Q = np.vstack([Q1.reshape(-1), Q2.reshape(-1)]).T
    out = potential.predict(Q).reshape(Q1.shape)

    all = np.vstack(
        c[:, [1, 0]] / np.array([out.shape[0], out.shape[1]]) * (bounds[:, 1] - bounds[:, 0]).T + bounds[:, 0]
        for c in find_contours(out - U0, dU)
    )

    every_nth = int(all.shape[0] / n)
    selection = all[::every_nth, :]
    if selection.shape[0] > n:
        return selection[:-1, :]
    else:
        return selection
