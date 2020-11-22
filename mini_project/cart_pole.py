import numpy as np
import sympy as sp
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import use

use('TkAgg')


def to_string(expression):
    # must replace x and theta at the end since we don't want to replace them within the derivative expressions
    return str(expression) \
        .replace('Derivative(theta(t), t)', 'theta_dot') \
        .replace('Derivative(theta(t), (t, 2))', 'theta_dot_dot') \
        .replace('Derivative(x(t), t)', 'x_dot') \
        .replace('Derivative(x(t), (t, 2))', 'x_dot_dot') \
        .replace('x(t)', 'x') \
        .replace('theta(t)', 'theta')


def get_equations_of_motion(t, x, theta, f, M, m, L, b, g):
    F_external = f - b * sp.diff(x, t)  # dissipative and external forces for the x direction

    x_rel = -(L / 2) * sp.sin(theta)  # COM of pole relative to cart
    v_rel = sp.diff(x_rel, t)
    v_rod_CM = sp.diff(x, t) + v_rel

    I_rod_CM = (m * L ** 2) / 12
    KE_cart = (M * sp.diff(x, t) ** 2) / 2
    KE_rod = (I_rod_CM * sp.diff(theta, t) ** 2) / 2 + (m * v_rod_CM ** 2) / 2
    KE = KE_cart + KE_rod
    PE = m * g * L * sp.cos(theta) / 2
    L = KE - PE

    # Lagrange's equations of the second kind
    # See this link for explanation of incorporating dissipative and external forces into Lagrangian mechanics:
    # https://physics.stackexchange.com/questions/518045/lagrangian-for-a-forced-system
    eq_1 = sp.Eq(sp.diff(sp.diff(L, sp.diff(x)), t), sp.diff(L, x) + F_external)
    eq_2 = sp.Eq(sp.diff(sp.diff(L, sp.diff(theta)), t), sp.diff(L, theta))

    result = sp.solve([eq_1, eq_2], [sp.diff(x, (t, 2)), sp.diff(theta, (t, 2))])
    x_dot_dot = result[sp.diff(x, (t, 2))]
    theta_dot_dot = result[sp.diff(theta, (t, 2))]

    # attempt any simplification
    x_dot_dot = sp.simplify(x_dot_dot)
    theta_dot_dot = sp.simplify(theta_dot_dot)

    return x_dot_dot, theta_dot_dot


def linearize(t, x, theta, f, expression):
    operating_point = {
        x: 0,
        sp.diff(x, t): 0,
        sp.diff(x, (t, 2)): 0,
        theta: 0,
        sp.diff(theta, t): 0,
        sp.diff(theta, (t, 2)): 0,
        f: 0
    }

    return sum([sp.diff(expression, var).subs(operating_point) * (var - operating_point[var])
                for var in [x, sp.diff(x, t), sp.diff(x, (t, 2)),
                            theta, sp.diff(theta, t), sp.diff(theta, (t, 2)),
                            f]])


def get_transfer_functions(t, x, theta, f, x_dot_dot, theta_dot_dot, s, X, Theta, F):
    laplace_tf = {
        x: X,
        sp.diff(x, t): s * X,
        sp.diff(x, (t, 2)): s ** 2 * X,
        theta: Theta,
        sp.diff(theta, t): s * Theta,
        sp.diff(theta, (t, 2)): s ** 2 * Theta,
        f: F
    }

    eq1 = sp.Eq(sp.diff(x, (t, 2)), x_dot_dot).subs(laplace_tf)
    eq2 = sp.Eq(sp.diff(theta, (t, 2)), theta_dot_dot).subs(laplace_tf)
    result = sp.solve([eq1, eq2], [X, Theta])
    X_tf = result[X] / F
    Theta_tf = result[Theta] / F

    # attempt partial fraction decomposition
    X_tf = sp.collect(sp.expand(X_tf), s)
    Theta_tf = sp.collect(sp.expand(Theta_tf), s)

    return X_tf, Theta_tf


def get_controller(t, x, theta, f, x_dot_dot, theta_dot_dot):
    """
    Values must be substituted in.
    """
    A = np.array([[0, 1, 0, 0],
                  [sp.diff(x_dot_dot, var) for var in [x, sp.diff(x, t), theta, sp.diff(theta, t)]],
                  [0, 0, 0, 1],
                  [sp.diff(theta_dot_dot, var) for var in [x, sp.diff(x, t), theta, sp.diff(theta, t)]]],
                 dtype=float)
    B = np.array([0, sp.diff(x_dot_dot, f), 0, sp.diff(theta_dot_dot, f)], dtype=float).reshape((4, 1))
    Q = np.identity(4)
    R = np.identity(1)
    return linear_quadratic_regulator(A, B, Q, R)


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


def test_controller(t, x, theta, f, x_dot_dot, theta_dot_dot, K, state_0=None, t_f=10):
    """
    Values must be substituted in.
    """
    if state_0 is None:
        state_0 = [0, 0, np.radians(70), 0]  # recovery from being tilted 70 degrees
        # state_0 = [-15, 0, 0, 0]  # recovery from being offset horizontally by 15m
        # state_0 = [0, 5, 0, 0]  # recovery from being imparted a velocity of 5 m/s
        # state_0 = [0, 0, 0, 30]  # recovery from being imparted an angular velocity of 30 rad/s

    params = [[x, sp.diff(x, t), theta, sp.diff(theta, t)], f]
    x_dot_dot_func = sp.lambdify(params, x_dot_dot)
    theta_dot_dot_func = sp.lambdify(params, theta_dot_dot)

    def state_derivative(_, state):
        # compute control output based on LQR gains
        u = -np.dot(K, state)

        # calculate derivative based on unsimplified equations of motion
        x_dot_dot_val = x_dot_dot_func(state, u)[0]
        theta_dot_dot_val = theta_dot_dot_func(state, u)[0]
        return np.array([state[1], x_dot_dot_val, state[3], theta_dot_dot_val])

    result = solve_ivp(state_derivative, (0, t_f), state_0, method='RK45', rtol=1e-6)
    # noinspection PyUnresolvedReferences
    return result.t, result.y


def main():
    t = sp.symbols('t')
    x = sp.Function('x')(t)
    theta = sp.Function('theta')(t)
    f = sp.Function('f')(t)
    M, m, L, b, g = sp.symbols('M, m, L, b, g')

    # get equations of motion
    x_dot_dot, theta_dot_dot = get_equations_of_motion(t, x, theta, f, M, m, L, b, g)
    print('x_dot_dot:', to_string(x_dot_dot))
    print('theta_dot_dot:', to_string(theta_dot_dot))

    # linearize
    x_dot_dot_linear = linearize(t, x, theta, f, x_dot_dot)
    theta_dot_dot_linear = linearize(t, x, theta, f, theta_dot_dot)
    print('x_dot_dot_linear: ', to_string(x_dot_dot_linear))
    print('theta_dot_dot_linear: ', to_string(theta_dot_dot_linear))

    # transfer functions
    s = sp.symbols('s')
    X = sp.Function('X')(s)
    Theta = sp.Function('Theta')(s)
    F = sp.Function('F')(s)
    X_tf, Theta_tf = get_transfer_functions(t, x, theta, f, x_dot_dot_linear, theta_dot_dot_linear, s, X, Theta, F)
    print('X_tf:', X_tf)
    print('Theta_tf:', Theta_tf)

    # substitute variables
    values = {
        M: 0.794,
        m: 0.00434,
        L: 0.185,
        b: 0.6113,
        g: 9.81
    }
    x_dot_dot = x_dot_dot.subs(values)
    theta_dot_dot = theta_dot_dot.subs(values)
    x_dot_dot_linear = x_dot_dot_linear.subs(values)
    theta_dot_dot_linear = theta_dot_dot_linear.subs(values)

    # create controller
    K = get_controller(t, x, theta, f, x_dot_dot_linear, theta_dot_dot_linear)
    print('K:', K)

    # graph
    t_vals, state_vals = test_controller(t, x, theta, f, x_dot_dot, theta_dot_dot, K)
    for i, label in enumerate(['x', 'x_dot', 'theta', 'theta_dot']):
        plt.plot(t_vals, state_vals[i, :], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Position (m), Velocity (m/s), $\theta$ (rad), $\omega$ (rad/s)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
