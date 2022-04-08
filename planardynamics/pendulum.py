import numpy as np
import scipy.integrate
from scipy.integrate import ode
from ..planarrobots import PlanarRobot
from .. import Progressbar


class StopIntegration(Exception):
    pass


def convergence_checker(pendulum, forget_ratio=.5):
    last = 10 * np.ones(pendulum.dof)

    def f(t, q, dq):
        nonlocal last
        filtered = (1 - forget_ratio) * last + forget_ratio * dq
        if np.linalg.norm(filtered) < 1e-4:
            return True

        last = filtered
        return False

    return f


def generate_bounds_callback(bounds):
    def cb(t, q, qdot):
        if q[0] < bounds[0][0] \
                or q[0] > bounds[0][1] \
                or q[1] < bounds[1][0] \
                or q[1] > bounds[1][1]:
            raise StopIntegration

    return cb


class Pendulum(PlanarRobot):
    def __init__(self, lengths, params, mass_fun, cc_fun, gravity_fun):
        super().__init__(link_lengths=lengths)
        self._mass_matrix = mass_fun
        self._coriolis_centrifugal = cc_fun
        self._gravity = gravity_fun
        self._params = params
        self._str = f"Pendulum with {super().number_of_joints} DoF. Parameters: {params}"
        self._torque_sources = []

    def add_torque_source(self, torque_source):
        self._torque_sources.append(torque_source)

    def _custom_torque(self, t, q, dq):
        if len(self._torque_sources) > 0:
            return np.sum([c(t, q, dq) for c in self._torque_sources], axis=0)
        else:
            return np.zeros(self.dof)

    @property
    def M(self):
        return self._mass_matrix

    @property
    def params(self):
        return self._params

    @property
    def C(self):
        return self._coriolis_centrifugal

    @property
    def G(self):
        return self._gravity

    def get_param(self, key):
        return self._params[key]

    def tau(self, q, dq, ddq):
        return self.M(q) @ ddq + self.C(q, dq) @ dq + self.G(q) - self._custom_torque(0.0, q, dq)

    def acc(self, q, dq):
        return np.linalg.inv(self.M(q)) @ (-self.C(q, dq) @ dq - self.G(q) + self._custom_torque(0.0, q, dq))

    def create_dynamics(self, controller=None, speed_factor=1.0):
        def dynamics(t, x):
            q, dq = x[0::2], x[1::2]
            M = self.M(q)
            C = self.C(q, dq)
            G = self.G(q)
            tau_int = self._custom_torque(t, q, dq)
            if controller is None:
                ddq = np.linalg.inv(M) @ (-C @ dq - G + tau_int)
            else:
                ddq = np.linalg.inv(M) @ (tau_int + controller(t, q, dq) - (C @ dq) - G)

            return speed_factor * np.ravel(np.column_stack((dq, ddq)))

        return dynamics

    def final_state(self, q0, dq0, t, controllers=None, speed_factor=1.0):
        def all_controllers(t, q, dq):
            if controllers:
                return np.sum([c(t, q, dq) for c in controllers], axis=0)
            else:
                return np.zeros_like(dq0)

        res = scipy.integrate.solve_ivp(
            fun=self.create_dynamics(controller=all_controllers, speed_factor=speed_factor),
            t_span=[0, t],
            y0=np.ravel(np.column_stack((q0, dq0))),
        )
        fs = res.y.T[-1, :]
        return fs[0::2], fs[1::2]

    def simit(self, q0, dq0, dt, controllers=None, dt_sigma=None):
        def all_controllers(t, q, dq):
            return np.sum([c(t, q, dq) for c in controllers], axis=0)

        x0 = np.ravel(np.column_stack((q0, dq0)))
        r = ode(self.create_dynamics(controller=all_controllers))
        r.set_integrator('dopri5')
        r.set_initial_value(x0)
        yield 0.0, x0
        while r.successful():
            dt_t = dt if dt_sigma is None else np.random.normal(loc=dt, scale=dt_sigma)
            if dt_t <= 0.0:
                dt_t = dt
            r.integrate(r.t + dt_t)
            yield r.t, r.y

    def sim(self, q0, dq0, dt,
            controller=None,
            t_max=100,
            dt_sigma=None,
            print_progress=False,
            controllers=None,
            stop_integration_observer=None,
            detect_cycle_closure=False,
            closure_threshold=1e-3,
            closure_speed_threshold=None,
            return_controller_history=False):
        if controller is not None:
            if controllers is None:
                controllers = [controller]
            else:
                controllers = [*controllers, controller]
        elif controllers is None:
            controllers = [lambda t, q, dq: np.zeros_like(q0)]

        progbar = Progressbar(t_max) if print_progress else None
        hist, thist, control_hist = [], [], []
        for t, state in self.simit(q0, dq0, dt, controllers=controllers, dt_sigma=dt_sigma):
            q, dq = state[0::2], state[1::2]
            if t > t_max:
                break
            if print_progress:
                progbar(t)
            if stop_integration_observer:
                if stop_integration_observer(t, q, dq):
                    break

            if detect_cycle_closure:
                if t > 0.1:
                    if np.linalg.norm(q - q0) < closure_threshold:
                        if closure_speed_threshold:
                            if np.linalg.norm(dq) < closure_speed_threshold:
                                break
                        else:
                            break

            hist.append(state)
            thist.append(t)
            if return_controller_history:
                control_hist.append(np.sum([c(t, q, dq) for c in controllers], axis=0))

        if return_controller_history:
            return np.array(thist), np.array(hist), np.array(control_hist)
        else:
            return np.array(thist), np.array(hist), None

    def kinetic_energy(self, q, qdot):
        if len(q.shape) == 1 and len(qdot.shape) == 1:
            return .5 * qdot.T @ self.M(q) @ qdot
        elif len(q.shape) == 2 and len(qdot.shape) == 2:
            m = np.empty((q.shape[0], self.number_of_joints, self.number_of_joints))
            for i in range(q.shape[0]):
                m[i, :, :] = self.M(q[i, :])

            return .5 * np.einsum("ni,nij,nj->n", qdot, m, qdot)
        else:
            raise ValueError

    def cartesian_velocity(self, q, qdot):
        jac = self.jacobian(q)
        return np.einsum("nij,nj->ni", jac, qdot)

    def __str__(self):
        return self._str
