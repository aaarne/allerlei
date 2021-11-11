import numpy as np
from functools import reduce
from .. import TUMColors


class RobotPlot:
    def __init__(self, ax=None, f=None, endeff_size=0.2):
        if ax is None:
            import matplotlib.pyplot as plt
            self.f, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.f = f
        size = endeff_size
        self.endeff_points = np.array([
            [size, -.5 * size, 1],
            [0, -.5 * size, 1],
            [0, .5 * size, 1],
            [size, .5 * size, 1]
        ]).T

    def plot_endeffector(self, pose, color='b', **kwargs):
        return self.ax.plot(*self._endeffector_lines(pose), c=color, lw=2, **kwargs)

    def _endeffector_lines(self, pose):
        transformed_points = (pose @ self.endeff_points).T
        return transformed_points[:, 0], transformed_points[:, 1]

    def animated_trajectory(self, robot, traj, t=None, color=TUMColors.TUMBlue, othercolor=TUMColors.TUMOrange,
                            interval=20, matthew=False, streamer=None, store_every_nth=None, text_function=None):
        lim = robot.forward_kinematics(np.zeros(robot.number_of_joints))[0, 0]
        lim *= 5 / 4
        self.ax.scatter(0, 0, c=othercolor)
        self.ax.axis('equal')
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))

        if matthew:
            import socket, struct
            UDP_IP = "127.0.0.1"
            UDP_PORT = 2222
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            stream_fcn = lambda q: sock.sendto(struct.pack("ddd", *q), (UDP_IP, UDP_PORT))
        elif streamer:
            stream_fcn = streamer
        else:
            stream_fcn = lambda q: q

        if t is not None or text_function is not None:
            time_info = self.ax.annotate('t=0', xy=np.array([-lim*0.5, lim*0.5]))

        lines = [self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)[0] for p1, p2 in
                 self._create_line_segments(robot, traj[0, :])]
        scatters = [self.ax.scatter(p2[0], p2[1], c=othercolor) for p1, p2 in
                    self._create_line_segments(robot, traj[0, :])]
        endeff_line = self.plot_endeffector(robot.endeffector_pose(traj[0, :]), color=color)[0]

        def animate(i):
            updated = list()
            stream_fcn(traj[i, :])
            if text_function is not None:
                time_info.set_text(text_function(i))
                updated.append(time_info)
            elif t is not None:
                time_info.set_text(f"t = {t[i]:.3f}s")
                updated.append(time_info)
            for line, scatter, (p1, p2) in zip(lines, scatters, self._create_line_segments(robot, traj[i, :])):
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                scatter.set_offsets([p2[0], p2[1]])
            endeff_line.set_data(*self._endeffector_lines(robot.endeffector_pose(traj[i, :])))

            if store_every_nth is not None and i % store_every_nth == 0:
                self.plot_robot(robot, traj[i, :], endeffector=False, plot_jacobian=False)
                self.f.canvas.draw()

            updated.append(endeff_line)
            updated.extend(lines)
            updated.extend(scatters)
            return updated

        from matplotlib.animation import FuncAnimation
        return FuncAnimation(self.f, animate, frames=traj.shape[0], interval=interval, blit=store_every_nth is None)

    def create_live_plot(self, robot, q0, color='b', othercolor='orange'):
        lim = robot.forward_kinematics(np.zeros(robot.number_of_joints))[0, 0]
        lim *= 5 / 4
        self.ax.scatter(0, 0, c=othercolor)
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))
        self.ax.grid()

        lines = [self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)[0] for p1, p2 in
                 self._create_line_segments(robot, q0)]
        scatters = [self.ax.scatter(p2[0], p2[1], c=othercolor) for p1, p2 in self._create_line_segments(robot, q0)]
        endeff_line = self.plot_endeffector(robot.endeffector_pose(q0), color=color)[0]

        def update(q):
            for line, scatter, (p1, p2) in zip(lines, scatters, self._create_line_segments(robot, q)):
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                scatter.set_offsets([p2[0], p2[1]])
            endeff_line.set_data(*self._endeffector_lines(robot.endeffector_pose(q)))
            if self.f is not None:
                self.f.canvas.draw()

        return update

    def _create_line_segments(self, robot, q):
        t = list(robot.link_trafos(q))

        def get_point_on_chain(i):
            trafo = reduce(lambda x, y: x @ y, t[0:i], np.eye(3)[np.newaxis, :, :])
            return trafo[0, 0:2, 2]

        for i in range(len(t)):
            yield get_point_on_chain(i), get_point_on_chain(i + 1)

    def plot_robot(self, robot, q, color=None, other_color=None, plot_jacobian=False, equal_aspect=True, dot_size=50,
                   manual_lim=None, full_lim=False, exclude_arm=False, endeffector=True, **kwargs):
        if color is None:
            color = TUMColors.TUMBlue
            if other_color is None:
                other_color = TUMColors.TUMOrange
        else:
            if other_color is None:
                other_color = color
        if not exclude_arm:
            for p1, p2 in self._create_line_segments(robot, q):
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color, **kwargs)
                self.ax.scatter(p2[0], p2[1], c=other_color, s=dot_size)

        if manual_lim is None:
            if full_lim:
                lim = robot.forward_kinematics(np.zeros(robot.number_of_joints))[0, 0]
                lim *= 5 / 4
                self.ax.set_xlim((-lim, lim))
                self.ax.set_ylim((-lim, lim))
            else:
                lim_min = np.min([self.ax.get_xlim()[0], self.ax.get_ylim()[0]])
                lim_max = np.max([self.ax.get_xlim()[1], self.ax.get_ylim()[1]])
                offset = 1 / 6 * (lim_max - lim_min)
                # self.ax.set_xlim((lim_min - offset, lim_max + offset))
                # self.ax.set_ylim((lim_min - offset, lim_max + offset))
        else:
            self.ax.set_xlim(manual_lim[0])
            self.ax.set_ylim(manual_lim[1])

        if equal_aspect:
            self.ax.set_aspect('equal')
        self.ax.scatter(0, 0, c=other_color, s=dot_size)
        if endeffector:
            self.plot_endeffector(robot.endeffector_pose(q), color=other_color, **kwargs)

        if plot_jacobian:
            endeff = robot.endeffector_pose(q)
            J = robot.jacobian(q)
            ones = np.ones(J.shape[1])
            self.ax.quiver(
                ones * endeff[0, 0, 2],
                ones * endeff[0, 1, 2],
                J[0, :],
                J[1, :],
                scale=10
            )

        self.ax.grid(b=True)


class RobotPlot2D(RobotPlot):
    def __init__(self, ax, base_interval=2 * np.pi):
        super(RobotPlot2D, self).__init__(ax)
        self.base_interval = base_interval

    def _evaluate_on_grid(self, robot, res=1000, dim=0):
        q1 = np.linspace(-.5 * self.base_interval, .5 * self.base_interval, res)
        q2 = np.linspace(-.5 * self.base_interval, .5 * self.base_interval, res)
        Q1, Q2 = np.meshgrid(q1, q2)
        Q = np.vstack([Q1.reshape(-1), Q2.reshape(-1)]).T

        out = robot.forward_kinematics(Q)[:, dim]

        Z = out.reshape(Q1.shape)

        return Q1, Q2, Z

    def contour(self, robot, dim=0, res=1000, clabel=True, levels=np.linspace(-1.5, 1.5, 7), **kwargs):
        Q1, Q2, Z = self._evaluate_on_grid(robot, res=res, dim=dim)

        cs = self.ax.contour(Q1, Q2, Z, levels=levels, **kwargs)
        if clabel:
            self.ax.clabel(cs)
        self.ax.grid()
        self.ax.set_xlabel('\$q_1\$')
        self.ax.set_ylabel('\$q_2\$')
        return cs

    def image(self, robot, dim=0, res=1000, cmap='jet', **kwargs):
        Q1, Q2, Z = self._evaluate_on_grid(robot, res=res, dim=dim)

        self.ax.set_xlabel('$q_1$')
        self.ax.set_ylabel('$q_2$')
        return self.ax.imshow(Z,
                              origin='lower',
                              cmap=cmap,
                              extent=[np.min(Q1), np.max(Q1), np.min(Q2), np.max(Q2)],
                              **kwargs)

    def manipulability(self, robot, res=1000, **kwargs):
        q1 = np.linspace(-.5 * self.base_interval, .5 * self.base_interval, res)
        q2 = np.linspace(-.5 * self.base_interval, .5 * self.base_interval, res)
        Q1, Q2 = np.meshgrid(q1, q2)
        Q = np.vstack([Q1.reshape(-1), Q2.reshape(-1)]).T

        out = robot.jacobian(Q)

        J = np.reshape(out, (Q1.shape[0], Q1.shape[1], 3, 2))[:, :, 0, :]
        M = np.linalg.norm(J, axis=2)

        self.ax.set_xlabel('$q_1$')
        self.ax.set_ylabel('$q_2$')
        return self.ax.imshow(M,
                              origin='lower',
                              cmap='jet',
                              extent=[np.min(q1), np.max(q1), np.min(q2), np.max(q2)],
                              **kwargs)

    def quiver(self, robot, eucliden=True, dim=0, res=50, scale=40, pivot='mid', width=5e-3, headwidth=3.0, angles='xy', **kwargs):
        q1 = np.linspace(-.5 * self.base_interval, .5 * self.base_interval, res)
        q2 = np.linspace(-.5 * self.base_interval, .5 * self.base_interval, res)
        Q1, Q2 = np.meshgrid(q1, q2)
        U, V = np.zeros_like(Q1), np.zeros_like(Q1)

        for i in range(Q1.shape[0]):
            for j in range(Q1.shape[1]):
                q = np.array([Q1[i, j], Q2[i, j]])
                J = robot.jacobian(q)
                if not eucliden:
                    mass = robot.mass(q)
                    J = J @ np.linalg.inv(mass)
                U[i, j] = J[dim, 0]
                V[i, j] = J[dim, 1]

        self.ax.quiver(Q1, Q2, U, V, pivot=pivot, headwidth=headwidth, scale=scale, width=width, angles=angles, **kwargs)
