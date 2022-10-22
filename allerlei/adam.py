import numpy as np

def create_adam(x0, alpha=1e-3, beta1=0.9, beta2=0.999):
    """Creates an Adam optimizer
        Returns a function object f. Use like this:
        loop:
            gradient = ...
            x += f(gradient)
    """
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)

    def adam(gradient):
        nonlocal m, v
        m = (beta1*m + (1-beta1)*gradient)/(1-beta1)
        v = (beta2*v + (1-beta2)*gradient*gradient)/(1-beta2)
        return alpha*m/(1e-8 + np.sqrt(v))

    return adam

