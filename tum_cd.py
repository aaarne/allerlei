import numpy as np


class TUMColor(str):
    def __init__(self, hexstring):
        super().__init__()
        self.hex = hexstring

    def __str__(self):
        return self.hex

    def __repr__(self):
        return self.hex

    def __hex__(self):
        return int(self.hex[1:], 16)

    def __len__(self):
        return len(self.hex)

    def to_numpy(self):
        return TUMColors.hextofloats(self.hex)

    def create_colormap(self):
        from matplotlib.colors import ListedColormap
        white = np.array([1, 1, 1])
        return ListedColormap(np.linspace(self.f, white, 256))

    def alpha(self, alpha):
        c = np.zeros(4)
        c[0:3] = self.f
        c[3] = alpha
        return c

    @property
    def cmap(self):
        return self.create_colormap()

    @property
    def f(self):
        return self.to_numpy()

    @property
    def f4(self):
        c = np.empty(4)
        c[0:3] = self.f
        c[3] = 1


class TUMColors:
    TUMBlue = TUMColor("#0065bd")
    TUMSecondaryBlue = TUMColor('#005293')
    TUMSecondaryBlue2 = TUMColor('#003359')
    TUMBlack = TUMColor('#000000')
    TUMWhite = TUMColor('#FFFFFF')
    TUMDarkGray = TUMColor('#333333')
    TUMGray = TUMColor('#808080')
    TUMLightGray = TUMColor('#CCCCC6')
    TUMAccentGray = TUMColor('#DAD7CB')
    TUMOrange = TUMColor("#e37222")
    TUMAccentOrange = TUMOrange
    TUMAccentGreen = TUMColor('#A2AD00')
    TUMAccentLightBlue = TUMColor('#98C6EA')
    TUMAccentBlue = TUMColor('#64A0C8')
    LightBlue = TUMColor('#3384ca')

    @staticmethod
    def hextofloats(h):
        return np.array([int(h[i:i + 2], 16) / 255. for i in (1, 3, 5)])

    @classmethod
    def cycle(cls):
        import itertools
        return map(lambda c: c.f, itertools.cycle([
            cls.TUMBlue,
            cls.TUMOrange,
            cls.TUMDarkGray,
            cls.TUMAccentLightBlue,
            cls.TUMAccentGreen,
        ]))

    @staticmethod
    def create_diverging_colormap(middle):
        from matplotlib.colors import ListedColormap
        orange = TUMColors.TUMOrange.f
        blue = TUMColors.TUMBlue.f
        newcolors = np.vstack((np.linspace(orange, middle, 128), np.linspace(middle, blue, 128)))
        newcmp = ListedColormap(newcolors[::-1], name="TUM")
        return newcmp

    @classmethod
    def create_cyclic_colormap(cls, repeated=1):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list('TUMcyclic', repeated*[
            cls.TUMWhite.f,
            cls.TUMOrange.f,
            cls.TUMBlack.f,
            cls.TUMBlue.f,
            cls.TUMWhite.f,
        ])

    @classmethod
    def create_cyclic_colormap_reversed(cls, repeated=1):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list('TUMcyclic', repeated*[
            cls.TUMDarkGray.f,
            cls.TUMBlue.f,
            cls.TUMWhite.f,
            cls.TUMOrange.f,
            cls.TUMDarkGray.f,
        ])

TUMBlue = TUMColors.TUMBlue
TUMOrange = TUMColors.TUMOrange
TUMDivergingColormap = TUMColors.create_diverging_colormap(np.array([1, 1, 1]))
TUMDivergingColormapDark = TUMColors.create_diverging_colormap(np.array([0, 0, 0]))
TUMDivergingColormapGrayish = TUMColors.create_diverging_colormap(np.array([.7, .7, .7]))
TUMCyclicColormap = TUMColors.create_cyclic_colormap()
