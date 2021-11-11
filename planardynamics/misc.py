import numpy as np
from numpy import deg2rad as r


def sample_phi(n, mu1=r(120), mu2=r(300), sigma=r(37)):
    u = np.random.uniform(low=0, high=1.0, size=n)
    phi = np.empty(n)
    n_cluster1 = np.count_nonzero(u < 0.5)
    phi[u < .5] = np.random.normal(loc=mu1, scale=sigma, size=n_cluster1)
    phi[u >= .5] = np.random.normal(loc=mu2, scale=sigma, size=n-n_cluster1)
    return phi


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    samples = sample_phi(100000)
    plt.hist(samples, bins=500)
    plt.show()