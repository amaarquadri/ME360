import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from mini_project.cart_pole import to_string, linear_quadratic_regulator
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')


def get_equations_of_motion(t, x, theta, f, M, m, L, b, g):
    F_external = f - b * sp.diff(x, t)  # dissipative and external forces for the x direction

    x_rel = -L * sp.sin(theta)
    v_x_rel = sp.diff(x_rel, t)
    v_x_pole = sp.diff(x, t) + v_x_rel

    y_rel = L * sp.cos(theta)
    v_y_pole = sp.diff(y_rel, t)  # relative and absolute velocities are the same

    KE_cart = (M * sp.diff(x, t) ** 2) / 2
    KE_pole = (m * v_x_pole ** 2) / 2 + (m * v_y_pole ** 2) / 2
    KE = KE_cart + KE_pole
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


def linearize(t, x, theta, f, expression, linearize_at_pi=False):
    operating_point = {
        x: 0,
        sp.diff(x, t): 0,
        sp.diff(x, (t, 2)): 0,
        theta: sp.pi if linearize_at_pi else 0,
        sp.diff(theta, t): 0,
        sp.diff(theta, (t, 2)): 0,
        f: 0
    }

    return expression.subs(operating_point) + sum(
        [sp.diff(expression, var).subs(operating_point) * (var - operating_point[var])
         for var in [x, sp.diff(x, t), sp.diff(x, (t, 2)), theta, sp.diff(theta, t), sp.diff(theta, (t, 2)), f]])


def get_transfer_functions(t, x, theta, f, x_dot_dot, theta_dot_dot, s, X, Theta, F, linearize_at_pi=False):
    laplace_tf = {
        x: X,
        sp.diff(x, t): s * X,
        sp.diff(x, (t, 2)): s ** 2 * X,
        theta: Theta + sp.pi if linearize_at_pi else Theta,
        sp.diff(theta, t): s * Theta,
        sp.diff(theta, (t, 2)): s ** 2 * Theta,
        f: F,
        sp.pi: sp.pi / s  # dirty fix for when theta = pi
    }

    eq1 = sp.Eq(sp.diff(x, (t, 2)), x_dot_dot).subs(laplace_tf)
    eq2 = sp.Eq(sp.diff(theta, (t, 2)), theta_dot_dot).subs(laplace_tf)
    result = sp.solve([eq1, eq2], [X, Theta])
    X_tf = result[X] / F
    Theta_tf = result[Theta] / F

    # attempt any simplification
    X_tf = sp.expand(sp.simplify(X_tf))
    Theta_tf = sp.expand(sp.simplify(Theta_tf))

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
    Q = np.diag([1, 2, 10, 5])
    R = np.array([[0.1]])
    return linear_quadratic_regulator(A, B, Q, R)


def test_controller(t, x, theta, f, x_dot_dot, theta_dot_dot, K, x_r_func=None, state_0=None, t_f=10):
    """
    Values must be substituted in.
    """
    if state_0 is None:
        # state_0 = [0, 0, np.radians(70), 0]  # recovery from being tilted 70 degrees
        # state_0 = [-0.1, 0, 0, 0]  # recovery from being offset horizontally by 15m
        # state_0 = [0, 5, 0, 0]  # recovery from being imparted a velocity of 5 m/s
        # state_0 = [0, 0, 0, 30]  # recovery from being imparted an angular velocity of 30 rad/s
        state_0 = [0, 0, 0, 0]
    if x_r_func is None:
        x_r_func = lambda t_: 0

    params = [[x, sp.diff(x, t), theta, sp.diff(theta, t)], f]
    x_dot_dot_func = sp.lambdify(params, x_dot_dot)
    theta_dot_dot_func = sp.lambdify(params, theta_dot_dot)

    def state_derivative(t_, state):
        # compute control output based on LQR gains
        u = np.dot(K, np.array([x_r_func(t_), 0, 0, 0]) - state)

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
        M: 1.994376,
        m: 0.105425,
        L: 0.110996,
        b: 1.6359,
        g: 9.81
    }
    x_dot_dot_linear = x_dot_dot_linear.subs(values)
    theta_dot_dot_linear = theta_dot_dot_linear.subs(values)
    x_dot_dot = x_dot_dot.subs(values)
    theta_dot_dot = theta_dot_dot.subs(values)

    # create controller
    K = get_controller(t, x, theta, f, x_dot_dot_linear, theta_dot_dot_linear)
    print('K:', K)

    # graph
    x_r_func = lambda t_: 1
    t_vals, state_vals = test_controller(t, x, theta, f, x_dot_dot, theta_dot_dot, K, x_r_func, t_f=30)
    # plt.plot(t_vals, x_r_func(t_vals), label=r'$x_{r}$')
    for i, label in enumerate(['x', 'v', r'$\theta$', r'$\omega$']):
        plt.plot(t_vals, state_vals[i, :], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Position (m), Velocity (m/s), $\theta$ (rad), $\omega$ (rad/s)')
    plt.title(r'Recovery from x Perturbation of 15 Meters')
    plt.legend()
    # plt.savefig('x_perturbation.png')
    plt.show()