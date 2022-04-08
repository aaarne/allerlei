import numpy as np
from scipy.optimize import root, minimize, root_scalar
from allerlei.lines import ClosedCurve

def equipotential_line(n, E, fun, center=np.array([0, 0])):
    points = np.zeros((n, 2))
    phi = np.linspace(0, 2*np.pi, n)
    for i in range(n):
        v = np.array([np.cos(phi[i]), np.sin(phi[i])])
        sol = minimize(lambda x: (fun(center + x*v) - E)**2, x0=0.1)
        points[i, :] = center + sol.x * v
    return ClosedCurve(points)

