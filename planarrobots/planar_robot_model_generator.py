from sympy import *
from functools import reduce


def _create_trafo(q, l):
    return Matrix([
        [cos(q), -sin(q), l * cos(q)],
        [sin(q), cos(q), l * sin(q)],
        [0, 0, 1]
    ])


def _print_code(expr):
    code = str(expr)
    code = code.replace("Matrix", "np.array")
    print(code)


def create_code(dim, code_gen=True):
    def symbol_name_suffix(ind):
        if code_gen:
            return f'[{ind}]'
        else:
            return f'_{ind + 1}'

    q = [Symbol(f'q{symbol_name_suffix(i)}') for i in range(dim)]
    l = [Symbol(f'l{symbol_name_suffix(i)}') for i in range(dim)]
    trafos = [_create_trafo(qi, li) for qi, li in zip(q, l)]
    tcp = reduce(lambda x, y: x @ y, trafos)
    fkin = Matrix([
        tcp[0, 2],
        tcp[1, 2],
        sum(q)
    ])
    J = simplify(Matrix([diff(fkin, qi).T for qi in q]).T)

    return fkin, J, q


def _main(dim, mode='python'):
    fkin, J, _ = create_code(dim, code_gen=mode=='python')
    print("Forward Kinematics:")
    if mode == 'python':
        _print_code(fkin)
    else:
        print(latex(fkin))

    print("Jacobian:")
    if mode == 'python':
        _print_code(J)
    else:
        print(latex(J))



if __name__ == '__main__':
    from fire import Fire

    Fire(_main)
