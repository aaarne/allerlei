import numpy as np
from scipy.linalg import sqrtm, inv, pinv, LinAlgError
from warnings import warn


def transform_mass(mass, jacobian):
    scalar = jacobian.shape[0] == 1
    Minv_x = jacobian @ inv(mass) @ jacobian.T
    if scalar:
        return 1.0/Minv_x.item()
    else:
        try:
            return inv(Minv_x)
        except LinAlgError:
            warn("Mass is singular")
            return pinv(Minv_x)


def damping_design(mass, stiffness, damping_ratio):
    if type(mass) is float:
        return 2*damping_ratio*np.sqrt(mass * stiffness)
    elif type(mass) is int:
        return damping_design(float(mass), stiffness, damping_ratio)
    elif type(mass) is np.ndarray:
        if mass.size == 1:
            return damping_design(mass.item(), stiffness, damping_ratio)
        else:
            if type(damping_ratio) is float:
                Zeta = np.eye(mass.shape[0]) * damping_ratio
            elif type(damping_ratio) is np.ndarray:
                if len(damping_ratio.shape) == 1:
                    Zeta = np.diag(damping_ratio)
                else:
                    Zeta = damping_ratio
            else:
                Zeta = np.diag(damping_ratio)
            M1 = sqrtm(mass)
            K1 = sqrtm(stiffness)
            return (M1 @ K1 + K1 @ M1) * 0.7
            # return M1 @ Zeta @ K1 + K1 @ Zeta @ M1


def weighted_pinv(matrix, weighing_matrix):
    J = matrix
    A = weighing_matrix
    return A @ J.T @ inv(J @ A @ J.T)

