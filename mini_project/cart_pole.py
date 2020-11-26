import numpy as np
import sympy as sp
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import use
from sympy.physics.vector.printing import vlatex
use('TkAgg')


def to_string(expression):
    return vlatex(expression) \
        .replace(r'\operatorname{Theta}', r'\Theta')


def get_gain(s, ratio, zeros, poles, epsilon=0.1):
    # find a value sufficiently far from all poles and zeros
    i = 0
    while any([abs(i - zero) < epsilon for zero, _ in zeros.items()]) or \
            any([abs(i - pole) < epsilon for pole, _ in poles.items()]):
        i += 1

    gain = ratio.subs(s, i)
    gain *= np.product([(i - pole) ** multiplicity for pole, multiplicity in poles.items()])
    gain /= np.product([(i - zeros) ** multiplicity for zeros, multiplicity in zeros.items()])
    return sp.re(gain.evalf())


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

    x_vec = np.array([x, sp.diff(x, t), theta, sp.diff(theta, t)])
    u_vec = np.array([f])

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
    return sum([sp.diff(expression, var).subs(operating_point) * var for var in np.concatenate((x_vec, u_vec))])


def get_transfer_functions(t, x_vec, u_vec, equations_of_motion):
    s = sp.symbols('s')

    X_vec = np.array([sp.Function(x.name.capitalize())(s) for x in x_vec[::2]])
    U_vec = np.array([sp.Function(u.name.capitalize())(s) for u in u_vec])

    laplace_tf = {}
    for x, X in zip(x_vec[::2], X_vec):
        laplace_tf[x] = X
        laplace_tf[sp.diff(x)] = s * X
        laplace_tf[sp.diff(x, (t, 2))] = s ** 2 * X
    for u, U in zip(u_vec, U_vec):
        laplace_tf[u] = U

    laplace_equations = [sp.Eq(sp.diff(x, t), eq).subs(laplace_tf)
                         for x, eq in zip(x_vec[1::2], equations_of_motion[1::2])]
    result = sp.solve(laplace_equations, list(X_vec))  # need to convert to list because sympy doesn't support numpy

    laplace_transforms = [sp.simplify(result[X]) for X in X_vec]
    return s, X_vec, U_vec, laplace_transforms


def get_controller_transfer_functions(s, X_vec, U_vec, transfer_functions):
    k_p_mat = np.array([[sp.symbols(f'k_p_{U.name.lower()}_{X.name.lower()}') for X in X_vec] for U in U_vec])
    k_d_mat = np.array([[sp.symbols(f'k_d_{U.name.lower()}_{X.name.lower()}') for X in X_vec] for U in U_vec])

    X_r_vec = np.array([sp.Function(X.name + '_r')(s) for X in X_vec])

    U_control_vec = np.dot(k_p_mat, (X_r_vec - X_vec)) - np.dot(k_d_mat, s * X_vec)

    eqs = [sp.Eq(X, tf).subs(zip(U_vec, U_control_vec)) for X, tf in zip(X_vec, transfer_functions)]
    result = sp.solve(eqs, list(X_vec))  # need to convert to array because sympy doesn't support numpy
    control_tfs = [result[X] for X in X_vec]

    k_pd_mat = np.zeros((len(U_vec), 2 * len(X_vec))).astype(object)
    k_pd_mat[:, 0::2] = k_p_mat
    k_pd_mat[:, 1::2] = k_d_mat

    return k_pd_mat, X_r_vec, control_tfs


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
    P = solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.multi_dot([np.linalg.inv(R), B.T, P])
    return K


def analyze_controller(s, X_vec, X_r_vec, k_pd_mat, controller_transfer_functions, K):
    controller_transfer_functions = [tf.subs(zip(k_pd_mat.flatten(), K.flatten()))
                                     for tf in controller_transfer_functions]

    for X_r in X_r_vec:
        print(f'\nSetting all but {X_r} to zero:')
        # set all others in X_r_vec to zero
        tfs = [sp.simplify(tf.subs([(X_r_, 0) for X_r_ in X_r_vec if X_r_ is not X_r]) / X_r)
               for tf in controller_transfer_functions]
        for X, tf in zip(X_vec, tfs):
            numerator, denominator = tf.as_numer_denom()
            zeros = sp.roots(numerator, s)
            poles = sp.roots(denominator, s)
            gain = get_gain(s, tf, zeros, poles)

            zeros_description = ', '.join(
                [str(zero.evalf(6)) if multiplicity == 1 else f'{zero.evalf(6)} (x{multiplicity})'
                 for zero, multiplicity in zeros.items()])
            poles_description = ', '.join(
                [str(pole.evalf(6)) if multiplicity == 1 else f'{pole.evalf(6)} (x{multiplicity})'
                 for pole, multiplicity in poles.items()])
            print(f'{X}/{X_r}: Gain: {gain.evalf(6)}, Poles: {poles_description}, Zeros: {zeros_description}')


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
    print('For displaying LaTeX:', 'https://latex.codecogs.com/eqneditor/editor.php')

    def substitute_constants(expression):
        return np.vectorize(lambda v: v.subs(constants))(expression)

    t, x_vec, u_vec, lagrangian, F_external_vec = physics_func(constants.keys())
    print('Lagrangian:', to_string(lagrangian))
    print('\nExternal Forces:', to_string(F_external_vec))

    equations_of_motion = get_equations_of_motion(t, x_vec, lagrangian, F_external_vec)
    print('\nEquations of Motion:')
    for var, eq in zip(x_vec[1::2], equations_of_motion[1::2]):
        print(to_string(sp.Eq(sp.diff(var, t), eq)))

    # operating_point = {x_vec[2]: sp.pi}
    linearized_equations_of_motion = [linearize(t, x_vec, u_vec, eq, operating_point) for eq in equations_of_motion]
    print('\nLinearized Equations of Motion (in Primed Variables):')
    for var, eq in zip(x_vec[1::2], linearized_equations_of_motion[1::2]):
        print(to_string(sp.Eq(sp.diff(var, t), eq)))

    s, X_vec, U_vec, transfer_functions = get_transfer_functions(t, x_vec, u_vec, linearized_equations_of_motion)
    transfer_functions = [sp.expand(tf) for tf in transfer_functions]
    print('\nTransfer Functions (in Primed Variables):')
    for var, tf in zip(X_vec, transfer_functions):
        print(to_string(sp.Eq(var, tf)))

    k_pd_mat, X_r_vec, control_transfer_functions = \
        get_controller_transfer_functions(s, X_vec, U_vec, transfer_functions)
    print('\nControl Transfer Functions (in Primed Variables):')
    for var, tf in zip(X_vec, control_transfer_functions):
        print(to_string(sp.Eq(var, tf)))

    A, B = get_matrix_equations_of_motion(x_vec, u_vec, linearized_equations_of_motion)
    print('\nA (in Primed Variables):\n', to_string(A), '\nB (in Primed Variables):\n', to_string(B))

    A = substitute_constants(A).astype(float)
    B = substitute_constants(B).astype(float)
    if Q is None:
        Q = np.identity(len(x_vec))
    if R is None:
        R = np.identity(len(u_vec))
    K = linear_quadratic_regulator(A, B, Q, R)
    print('K:\n', K)

    analyze_controller(s, X_vec, X_r_vec, k_pd_mat, substitute_constants(control_transfer_functions), K)

    equations_of_motion = [substitute_constants(eq) for eq in equations_of_motion]
    t_vals, state_vals = test_controller(x_vec, u_vec, equations_of_motion, K, x_r_func=x_r_func, x_0=x_0, t_f=t_f)

    print('\nMax Force:', np.max(np.abs([np.dot(K, x_r_func(t_) - state_vals[:, i])[0]
                                         for i, t_ in enumerate(t_vals)])))

    for i, x in enumerate(x_vec):
        plt.plot(t_vals, state_vals[i, :], label=to_string(x))


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
