import numpy as np
import matplotlib.pyplot as plt
from ..planarrobots import RobotPlot

_animations = list()

def plot_pendulum_trajectory(pendulum, t, traj,
                             plot_orientation=True,
                             energy=True,
                             phase_plots=True,
                             plot_fkin=False,
                             auto_corr=False,
                             streamer=None,
                             **kwargs):
    labels = {
        0: '$q_1$',
        2: '$q_2$',
        1: '$\\omega_1$',
        3: '$\\omega_2$',
    }
    figures = []
    delta = traj[:, 0::2] - pendulum.equilibrium
    f, axes = plt.subplots(3, 1)
    axes[0].plot(t, traj[:, 0], label=labels[0])
    axes[0].plot(t, traj[:, 2], label=labels[2])
    axes[1].plot(t, delta[:, 0], label='$\\Delta q_1$')
    axes[1].plot(t, delta[:, 1], label='$\\Delta q_2$')
    axes[2].plot(t, traj[:, 1], label=labels[1])
    axes[2].plot(t, traj[:, 3], label=labels[3])
    for ax in axes:
        ax.grid()
        ax.legend()
    f.tight_layout()
    figures.append(f)

    if auto_corr:
        f, ax = plt.subplots()
        from scipy.signal import correlate
        ax.plot(correlate(delta[:, 0], delta[:, 0]))
        ax.plot(correlate(delta[:, 1], delta[:, 1]))

    if phase_plots:
        f, axes = plt.subplots(2, 2)
        for ax, a, b in [
            (axes[0, 0], 0, 2),
            (axes[0, 1], 1, 3),
            (axes[1, 0], 0, 1),
            (axes[1, 1], 2, 3),
        ]:
            ax.plot(traj[:, a], traj[:, b])
            ax.set_xlabel(labels[a])
            ax.set_ylabel(labels[b])
            ax.grid()
        f.tight_layout()
        figures.append(f)

    if plot_fkin:
        f, axt = plt.subplots()
        fkin = pendulum.forward_kinematics(traj[:, 0::2])
        handles = [
            axt.plot(t, fkin[:, 0], label='x'),
            axt.plot(t, fkin[:, 1], label='y'),
        ]
        if plot_orientation:
            axo = axt.twinx()
            handles.append(axo.plot(t, fkin[:, 2], '--', label='φ'))
        plt.legend(handles=[h[0] for h in handles])
        plt.grid()
        f.tight_layout()
        figures.append(f)

    if energy:
        f, ax = plt.subplots()
        K = pendulum.kinetic_energy(traj[:, 0::2], traj[:, 1::2])
        V = pendulum.potential_energy(traj[:, 0::2])
        E = K + V
        ax.plot(t, K, label='K')
        ax.plot(t, V, label='V')
        ax.plot(t, E, label='E')
        ax.grid()
        ax.legend()
        f.tight_layout()
        figures.append(f)

    f, ax = plt.subplots()
    link_fkin = pendulum.forward_kinematics_for_each_link(traj[:, 0::2])
    ax.plot(link_fkin[:, 1, 0], link_fkin[:, 1, 1], alpha=.3)
    ax.plot(link_fkin[:, 0, 0], link_fkin[:, 0, 1], alpha=.3)
    plot = RobotPlot(ax, f=f)
    anim = plot.animated_trajectory(pendulum, traj[:, 0::2], t=t, streamer=streamer)
    _animations.append(anim)
    figures.append(f)

    return figures
