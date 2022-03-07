import numpy as np
import matplotlib.lines as mlines
from matplotlib.patches import Patch


def create_hypercube(dim, resolution, exclude_end=False):
    cube = np.array([np.linspace(0, 1, resolution)]).T
    if exclude_end:
        cube = cube[0:-2]
    for i in range(dim - 1):
        n = np.linspace(0, 1, resolution)
        tiled = np.tile(cube, (len(n), 1))
        repeated = np.repeat(n, cube.shape[0]).reshape((-1, 1))
        cube = np.hstack([repeated, tiled])
    return cube


def my_unwrap(a):
    for col in range(a.shape[1]):
        a[:, col] = np.arctan2(np.sin(a[:, col]), np.cos(a[:, col]))


def radify(angles):
    a = angles.copy()
    for col in range(a.shape[1]):
        a[:, col] = np.arctan2(np.sin(a[:, col]), np.cos(a[:, col]))
    return a


def angular_dist(q0, q1):
    return sum(
        ((np.sin(a) - np.sin(b)) ** 2 + (np.cos(a) - np.cos(b)) ** 2 for a, b in zip(q0, q1))
    )


def np_cartesian_product(a1, a2):
    return np.transpose([np.repeat(a1, len(a2)), np.tile(a2, len(a1))])


def print_statistics(e):
    for i, coor in zip([0, 1, 2], ['x', 'y', 'ϕ']):
        d = np.abs(e[:, i])
        print("{}-direction:\tmax: {:.6f}\taverage: {:.6f}\tstddev: {:.6f}".format(coor, np.max(d), np.average(d),
                                                                                   np.std(d)))


def print_progress(v):
    count = 100 * v
    bar_len = 100
    bar = '#' * int(round(bar_len * (count / 100.0))) + '-' * int(round(bar_len * 0.01 * (100 - count)))
    print(f'\rProgress: [{bar}]  {count:.1f}%', end='')


class Progressbar:
    def __init__(self, target=None, length=100, graceful=False):
        if target:
            self._target = float(target)
        else:
            self._target = None
        self._l = length
        self._grace = graceful

    def __call__(self, arg):
        if isinstance(arg, (float, int)):
            return self.print(arg)
        else:
            return self.wrap(arg)

    def __enter__(self):
        value = 0
        self(value)

        def update(amount=1):
            nonlocal value
            value += amount
            if not self._grace and value > (self._target if self._target else 100):
                raise ValueError("Percentage beyond 100%")
            self.print(value)

        return update

    def __exit__(self, *args):
        self(self._target if self._target else 100)
        print()

    def print(self, arg):
        if self._target:
            filled = arg / self._target
            empty = 1 - filled
            closing = f"{int(arg)}/{int(self._target)} ({(filled * 100.0):.1f}%)"
        else:
            filled = arg / 100.0
            empty = 1.0 - filled
            closing = f"{(filled * 100.0):.1f}%"

        bar = '#' * int(round(self._l * filled)) + '-' * int(round(self._l * empty))
        print(f'\rProgress: [{bar}] {closing}', end='')
        return filled

    def wrap(self, it):
        with self as f:
            for v in it:
                yield v
                f()



def progressify(iterable, n=None):
    if n is None:
        n = len(iterable)

    with Progressbar(n) as bar:
        for value in iterable:
            yield value
            bar()


def plot_histogram(e, bins=100):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3)

    for index, coor in enumerate(['x', 'y', 'φ']):
        ax = axes[index]
        ax.hist(e[:, index], bins=bins)
        ax.semilogy()
        ax.set_title("Error in {}-direction".format(coor))
        ax.grid()
        if index in [0, 1]:
            ax.set_xlabel("Deviation in {}-direction in [m]")
        else:
            ax.set_xlabel("Orientation deviation in rad")


def hom2xyphi(hom):
    fkin = np.empty((hom.shape[0], 3))
    fkin[:, 0] = hom[:, 0, 2]
    fkin[:, 1] = hom[:, 1, 2]
    fkin[:, 2] = np.angle(np.exp(1j * (np.arctan2(hom[:, 1, 0], hom[:, 0, 0]))))
    return fkin


def xyphi2hom(xyphi):
    x = xyphi[0]
    y = xyphi[1]
    phi = xyphi[2]
    return np.array([
        [np.cos(phi), -np.sin(phi), x],
        [np.sin(phi), np.cos(phi), y],
        [0, 0, 1]
    ])


def sample_joint_angles(dim):
    return np.random.uniform(0, 2 * np.pi, dim)


def to_color(array, name='viridis'):
    from matplotlib.cm import get_cmap
    cmap = get_cmap(name)
    from matplotlib.colors import Normalize
    value_normalizer = Normalize(vmin=np.min(array), vmax=np.max(array))
    return cmap(value_normalizer(array))


def tex():
    from matplotlib import rc
    rc('text', usetex=True)


def matlabstruct2dict(struct):
    if type(struct) is dict:
        return {k: matlabstruct2dict(v) for k, v in struct.items()}
    elif type(struct) in [bytes, str, int, float]:
        return struct
    elif type(struct) is list:
        return [*map(matlabstruct2dict, struct)]
    elif type(struct) is np.ndarray:
        if struct.dtype.names is not None:
            return {k: matlabstruct2dict(struct[k]) for k in struct.dtype.names}
        elif struct.dtype.hasobject:
            if struct.flatten().shape[0] > 1:
                return [matlabstruct2dict(s) for s in struct.flatten()]
            else:
                return matlabstruct2dict(struct.item())
        else:
            return struct
    else:
        raise ValueError


def unpack2(array):
    return [array[:, 0], array[:, 1]]


def unpack3(array):
    return [array[:, 0], array[:, 1], array[:, 2]]


def split_lines_on_torus(traj):
    diffs = np.max(np.abs(np.diff(traj, axis=0)), axis=1)
    indices = [0, *[i.item() + 1 for i in np.argwhere(diffs > np.pi)], traj.shape[0] - 1]
    for i, j in zip(indices, indices[1::]):
        yield traj[i:j]


def all_combinations(x, y):
    xx, yy = np.meshgrid(x, y)
    return np.vstack([xx.reshape(-1), yy.reshape(-1)]).T


class MyMeshgrid:
    def __init__(self, q1, q2):
        Q1, Q2 = np.meshgrid(q1, q2)
        self._Q = np.vstack([Q1.reshape(-1), Q2.reshape(-1)]).T
        self._Q1 = Q1
        self._Q2 = Q2

    def __iter__(self):
        return iter(self.Q)

    @property
    def Q(self):
        return self._Q

    @property
    def shape(self):
        return self._Q.shape

    @property
    def n(self):
        return self.shape[0]

    def to_plt_meshgrid(self, Q):
        return Q.reshape(self._Q1.shape)

    @property
    def extent(self):
        Q1, Q2 = self._Q1, self._Q2
        return [np.min(Q1), np.max(Q1), np.min(Q2), np.max(Q2)]

    def contour_args(self, Q):
        return self._Q1, self._Q2, self.to_plt_meshgrid(Q)

    @property
    def indices(self):
        return np.arange(0, self.n)


class LegendEntries:
    @staticmethod
    def line(c, l, **kwargs):
        return mlines.Line2D([0], [0], color=c, label=l, **kwargs)

    @staticmethod
    def scatter(color, marker, label, **kwargs):
        return mlines.Line2D([0], [0], linewidth=0, marker=marker, color=color, label=label, **kwargs)

    @staticmethod
    def patch(facecolor, label, **kwargs):
        return Patch(facecolor=facecolor, label=label, **kwargs)


