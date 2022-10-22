import numpy as np

def numerical_grad(f, x, dx=0.1):
    return (f(x+dx) - f(x)) / np.linalg.norm(dx)


def numerical_grad_centered(f, x, dx=0.1):
    return (f(x+dx) - f(x-dx)) / (2*np.linalg.norm(dx))


def five_point_stencil(f, x, dx):
    return (-f(x+2*dx) + 8*f(x + dx) - 8*f(x - dx) + f(x-2*dx)) / (12*np.linalg.norm(dx))


def numerical_jacobian(f, x, eps, method=numerical_grad_centered):
    base = f(x)
    result = np.zeros((base.shape[0], x.shape[0]))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            def partial_f(chi):
                cx = x.copy()
                cx[j] = chi
                return f(cx)[i]
            result[i, j] = method(partial_f, x[j], eps)

    return result
