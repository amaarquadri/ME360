import numpy as np
import sympy as sp
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')


def to_string(expression, var_strings=None, latex=False):
    ESCAPE_VARS = ['theta', 'Theta']  # append to this as necessary

    if var_strings is None:
        var_strings = ['x', 'theta', 'f']

    text = str(expression).replace('**', '^')
    for var_string in var_strings:
        new_var_string = f'\\{var_string}' if (latex and (var_string in ESCAPE_VARS)) else var_string

        # must replace var_string at the end since we don't want to replace it within the derivative expressions
        text = text \
            .replace(f'Derivative({var_string}(t), t)',
                     f'$\\dot{{{new_var_string}}}$' if latex else f'{new_var_string}_dot') \
            .replace(f'Derivative({var_string}(t), (t, 2))',
                     f'$\\ddot{{{new_var_string}}}$' if latex else f'{new_var_string}_dot_dot') \
            .replace(f'{var_string}(t)',
                     f'${new_var_string}$' if latex else new_var_string)
    return text


def get_cart_pole_physics(constants):
    M, m, L, b, g = constants

    t = sp.symbols('t')

    # define state variables
    x = sp.Function('x')(t)
    theta = sp.Function('theta')(t)

    # define control variables
    f = sp.Function('f')(t)

    x_rel = -L * sp.sin(theta)
    v_x_rel = sp.diff(x_rel, t)
    v_x_pole = sp.diff(x, t) + v_x_rel

    y_rel = L * sp.cos(theta)
    v_y_pole = sp.diff(y_rel, t)  # relative and absolute velocities are the same

    KE_cart = (M * sp.diff(x, t) ** 2) / 2
    KE_pole = (m * v_x_pole ** 2) / 2 + (m * v_y_pole ** 2) / 2
    KE = KE_cart + KE_pole
    PE = m * g * L * sp.cos(theta)
    L = sp.simplify(KE - PE)

    f_x_external = f - b * sp.diff(x, t)  # dissipative and external forces for the x direction
    f_theta_external = 0

    x_vec = [x, sp.diff(x, t), theta, sp.diff(theta, t)]
    u_vec = [f]

    return t, x_vec, u_vec, L, [f_x_external, f_theta_external]


def get_equations_of_motion(t, x_vec, L, F_external_vec):
    """
    The generalized coordinates should be in the even indices of x_vec
    and their corresponding derivatives should be in the odd indices.
    """
    lagrange_equations = []
    for x, F_external in zip(x_vec[::2], F_external_vec):
        lagrange_equations.append(sp.Eq(sp.diff(sp.diff(L, sp.diff(x, t)), t), sp.diff(L, x) + F_external))
    result = sp.solve(lagrange_equations, [sp.diff(x, (t, 2)) for x in x_vec[::2]])

    equations_of_motion = []
    for i, x in enumerate(x_vec[::2]):
        equations_of_motion.append(sp.diff(x, t))
        equations_of_motion.append(sp.simplify(result[sp.diff(x, (t, 2))]))

    return equations_of_motion


def linearize(t, x_vec, u_vec, expression, operating_point=None):
    if operating_point is None:
        operating_point = {}

    # set unspecified required values to 0
    for x in x_vec[::2]:
        if x not in operating_point:
            operating_point[x] = 0
        if sp.diff(x, t) not in operating_point:
            operating_point[sp.diff(x, t)] = 0
        if sp.diff(x, (t, 2)) not in operating_point:
            operating_point[sp.diff(x, (t, 2))] = 0
    for u in u_vec:
        if u not in operating_point:
            operating_point[u] = 0

    # The operating points are not included in the linearization which means the resulting linear system
    # (x_dot = A * x + B * u) is in reference to the primed coordinates.
    # Thus the K gains for the controller will also correspond to the primed coordinates.
    # However, K * (x_r' - x') = K * (x_r - x) so the controller works the same regardless.
    return sum([sp.diff(expression, var).subs(operating_point) * var for var in x_vec + u_vec])


def get_transfer_functions(t, x_vec, u_vec, equations_of_motion):
    s = sp.symbols('s')

    X_vec = [sp.Function(x.name.capitalize())(s) for x in x_vec[::2]]
    U_vec = [sp.Function(u.name.capitalize())(s) for u in u_vec]

    laplace_tf = {}
    for x, X in zip(x_vec[::2], X_vec):
        laplace_tf[x] = X
        laplace_tf[sp.diff(x)] = s * X
        laplace_tf[sp.diff(x, (t, 2))] = s ** 2 * X
    for u, U in zip(u_vec, U_vec):
        laplace_tf[u] = U

    laplace_equations = [sp.Eq(sp.diff(x, t), eq).subs(laplace_tf)
                         for x, eq in zip(x_vec[1::2], equations_of_motion[1::2])]
    result = sp.solve(laplace_equations, X_vec)

    laplace_transforms = [sp.simplify(result[X]) for X in X_vec]
    return s, X_vec, U_vec, laplace_transforms


def get_matrix_equations_of_motion(x_vec, u_vec, equations_of_motion):
    A = np.array([[sp.diff(eq, x) for x in x_vec] for eq in equations_of_motion])
    B = np.array([[sp.diff(eq, u) for u in u_vec] for eq in equations_of_motion])
    return A, B


def linear_quadratic_regulator(A, B, Q, R):
    """
    https://youtu.be/bMiiC94FJ5E?t=3276
    https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py
    System is defined by dx/dt = Ax + Bu
    Minimizing integral (x.T*Q*x + u.T*R*u) dt from 0 to infinity
    Returns K such that optimal control is u = -Kx
    """
    # first, try to solve the Ricatti equation
    S = solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.multi_dot([np.linalg.inv(R), B.T, S])
    return K


def test_controller(x_vec, u_vec, equations_of_motion, K, x_r_func=None, x_0=None, t_f=10):
    """
    A, B, K, and x_0 must be numpy arrays and x_r_func must be a function that returns a numpy array.
    """
    if x_0 is None:
        x_0 = np.zeros(len(equations_of_motion))
    if x_r_func is None:
        # use step input for first variable in x_r
        target = np.zeros(len(equations_of_motion))
        target[0] = 1

        def x_r_func(_):
            return target

    equations_of_motion = [sp.lambdify([x_vec, u_vec], eq) for eq in equations_of_motion]

    def state_derivative(t, x_vec_):
        u_vec_ = np.dot(K, x_r_func(t) - x_vec_)
        return np.array([eq(x_vec_, u_vec_) for eq in equations_of_motion])

    result = solve_ivp(state_derivative, (0, t_f), x_0, method='RK45', rtol=1e-6)
    # noinspection PyUnresolvedReferences
    return result.t, result.y


def evaluate_controller(x_vec, u_vec, equations_of_motion, K, x_r_func=None, x_0=None, t_f=10):
    pass


def build_and_test_controller(constants, physics_func=get_cart_pole_physics, operating_point=None, Q=None, R=None,
                              x_r_func=None, x_0=None, t_f=10):
    """
    Generates

    :param constants: A dictionary that maps sympy variables to floating point values.
                      The sympy variables must be in the order that physics_func expects.
    :param physics_func:
    :param operating_point:
    :param Q:
    :param R:
    :param x_r_func:
    :param x_0:
    :param t_f:
    """

    def substitute_constants(expression):
        return np.vectorize(lambda v: v.subs(constants))(expression)

    t, x_vec, u_vec, lagrangian, F_external_vec = physics_func(constants.keys())
    print('Lagrangian:', to_string(lagrangian))
    print('\nExternal Forces:', to_string(F_external_vec))

    equations_of_motion = get_equations_of_motion(t, x_vec, lagrangian, F_external_vec)
    print('\nEquations of Motion:')
    for var, eq in zip(x_vec, equations_of_motion):
        print(to_string(sp.diff(var, t)), '=', to_string(eq))

    # operating_point = {x_vec[2]: sp.pi}
    linearized_equations_of_motion = [linearize(t, x_vec, u_vec, eq, operating_point) for eq in equations_of_motion]
    print('\nLinearized Equations of Motion (in Primed Variables):')
    for var, eq in zip(x_vec, linearized_equations_of_motion):
        print(to_string(sp.diff(var, t)), '=', to_string(eq))

    s, X_vec, U_vec, transfer_functions = get_transfer_functions(t, x_vec, u_vec, linearized_equations_of_motion)
    transfer_functions = [sp.expand(tf) for tf in transfer_functions]
    print('\nTransfer Functions (in Primed Variables):')
    for var, tf in zip(X_vec, transfer_functions):
        print(to_string(var), '=', to_string(tf))

    A, B = get_matrix_equations_of_motion(x_vec, u_vec, linearized_equations_of_motion)
    print('\nA (in Primed Variables):\n', A, '\nB (in Primed Variables):\n', B)

    A = substitute_constants(A).astype(float)
    B = substitute_constants(B).astype(float)
    if Q is None:
        Q = np.identity(len(x_vec))
    if R is None:
        R = np.identity(len(u_vec))
    K = linear_quadratic_regulator(A, B, Q, R)
    print('K:\n', K)

    equations_of_motion = [substitute_constants(eq) for eq in equations_of_motion]
    t_vals, state_vals = test_controller(x_vec, u_vec, equations_of_motion, K, x_r_func=x_r_func, x_0=x_0, t_f=t_f)

    print('\nMax Force:', np.max(np.abs([np.dot(K, x_r_func(t_) - state_vals[:, i])[0]
                                         for i, t_ in enumerate(t_vals)])))

    for i, x in enumerate(x_vec):
        plt.plot(t_vals, state_vals[i, :], label=to_string(x, latex=True))


def main():
    M, m, L, b, g = sp.symbols('M, m, L, b, g')
    constants = {
        M: 1.994376,
        m: 0.105425,
        L: 0.110996,
        b: 1.6359,
        g: 9.81
    }
    build_and_test_controller(constants, Q=np.diag([10, 20, 100, 50]), R=np.array([[1]]),
                              x_r_func=lambda t: np.zeros(4),
                              x_0=np.array([0, 0, np.radians(75), 0]))

    # plt.plot(t_vals, x_r_func(t_vals)[0], label=r'$x_{r}$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Position (m), Velocity (m/s), $\theta$ (rad), $\omega$ (rad/s)')
    plt.title(r'Recovery from x Perturbation of 15 Meters')
    plt.legend()
    # plt.savefig('x_perturbation.png')
    plt.show()


if __name__ == '__main__':
    main()
