import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')


"""
This code models the dynamics 
"""


def linear_quadratic_regulator(A, B, Q, R):
    """
    https://youtu.be/bMiiC94FJ5E?t=3276
    https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py
    System is defined by dx/dt = Ax + Bu\
    Minimizing integral (x.T*Q*x + u.T*R*u) dt from 0 to infinity
    Returns K such that optimal control is u = -Kx
    """
    # first, try to solve the ricatti equation
    S = solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.multi_dot([np.linalg.inv(R), B.T, S])
    return K


def get_equations_of_motion(t, x, theta, desired_theta_dot_dot, m_t=10, m_p=1, g=9.81, b=10, L=1):
    print(f'm_t={m_t}, m_p={m_p}, g={g}, b={b}, L={L}')
    u = sp.Function('u')(t)

    x_p = x + L * sp.sin(theta)
    y_p = -L * sp.cos(theta)

    T = (m_p * sp.diff(y_p, t, 2) + m_p * g) / sp.cos(theta)  # force balance in y direction for ball
    eq1 = sp.Eq(m_t * sp.diff(x, t, 2), u - b * sp.diff(x, t) + T * sp.sin(theta))  # force balance for cart
    eq2 = sp.Eq(m_p * sp.diff(x_p, t, 2), -T * sp.sin(theta))  # force balance in y direction for ball

    # determine equations of motion
    equations_of_motion = sp.solve([eq1, eq2], [sp.diff(x, t, 2), sp.diff(theta, t, 2)])
    x_dot_dot = equations_of_motion[sp.diff(x, t, 2)]
    theta_dot_dot = equations_of_motion[sp.diff(theta, t, 2)]

    # set u such that the pendulum dynamics reduce to the desired dynamics
    desired_u = sp.solve(sp.Eq(theta_dot_dot, desired_theta_dot_dot), u)[0]
    print(f'u(t)={desired_u}')

    return x_dot_dot.subs(u, desired_u), theta_dot_dot.subs(u, desired_u)


def get_state_derivative(t, x, theta, x_dot_dot, theta_dot_dot):
    params = [t, x, sp.diff(x, t), theta, sp.diff(theta)]
    x_dot_dot_func = sp.lambdify(params, x_dot_dot)
    theta_dot_dot_func = sp.lambdify(params, theta_dot_dot)

    def derivative(t_val, state):
        x_val, x_dot_val, theta_val, theta_dot_val = state
        x_dot_dot_val = x_dot_dot_func(t_val, x_val, x_dot_val, theta_val, theta_dot_val)
        theta_dot_dot_val = theta_dot_dot_func(t_val, x_val, x_dot_val, theta_val, theta_dot_val)
        return np.array([x_dot_val, x_dot_dot_val, theta_dot_val, theta_dot_dot_val])

    return derivative


def main():
    t = sp.symbols('t')
    x = sp.Function('x')(t)
    theta = sp.Function('theta')(t)
    desired_theta_dot_dot = -1.9 * sp.diff(theta) - theta + sp.pi
    x_dot_dot, theta_dot_dot = get_equations_of_motion(t, x, theta, desired_theta_dot_dot)
    state_derivative = get_state_derivative(t, x, theta, x_dot_dot, theta_dot_dot)

    result = solve_ivp(state_derivative, (0, 20), [13, 7, np.radians(45), -3], max_step=1e-2)
    print(result.message)

    fig, ax = plt.subplots(4, 1, sharex='col')
    for i, ylabel in zip(range(4), ['Position (m)', 'Velocity (m/s)', 'Angle (deg)', 'Angular Velocity (deg/s)']):
        ax[i].plot(result.t, result.y[i] if i < 2 else np.degrees(result.y[i]))
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel(ylabel)

    plt.show()


if __name__ == '__main__':
    main()
