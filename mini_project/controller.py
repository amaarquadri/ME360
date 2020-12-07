from abc import ABC, abstractmethod
import numpy as np
import sympy as sp
from tbcontrol.symbolic import routh
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from sympy.physics.vector.printing import vlatex
import matplotlib.pyplot as plt
from matplotlib import use

use('TkAgg')


class PhysicsSystem(ABC):
    def __init__(self, constants, operating_point=None, C=None, Q=None, R=None, V=None, W=None):
        self.constants = constants
        self.t = sp.symbols('t')

        self.x, self.u = self.create_x_and_u()
        self.x_dim, self.u_dim = len(self.x), len(self.u)
        self.lagrangian = self.create_lagrangian()
        self.f_external = self.create_f_external()

        # set unspecified required values to 0
        self.operating_point = operating_point if operating_point is not None else {}
        for x in self.x:
            if x not in self.operating_point:
                self.operating_point[x] = 0
            if sp.diff(x, self.t) not in self.operating_point:
                self.operating_point[sp.diff(x, self.t)] = 0
        for u in self.u:
            if u not in self.operating_point:
                self.operating_point[u] = 0

        self.s = sp.symbols('s')
        self.X = np.array([sp.Function(x.name.capitalize())(self.s) for x in self.x])
        self.U = np.array([sp.Function(u.name.capitalize())(self.s) for u in self.u])
        self.X_r = np.array([sp.Function(X.name + '_r')(self.s) for X in self.X])

        if C is not None:
            # TODO: what if C depends on constants and needs substitution
            self.C = C
        else:
            # measure just first coordinate by default
            self.C = np.zeros((1, 2 * self.x_dim))
            self.C[0, 0] = 1
        self.y_dim = self.C.shape[0]

        self.Q = Q if Q is not None else np.identity(2 * self.x_dim)
        self.R = R if R is not None else np.identity(self.u_dim)
        self.V = V if V is not None else np.identity(2 * self.x_dim)  # model noise
        self.W = W if W is not None else np.identity(self.y_dim)  # measurement noise

        self.numpy_substitutor = np.vectorize(lambda v: v if isinstance(v, float) or isinstance(v, int) else
                                              v.subs(self.constants))

    @abstractmethod
    def create_x_and_u(self):
        pass

    @abstractmethod
    def create_lagrangian(self):
        pass

    @abstractmethod
    def create_f_external(self):
        pass

    def apply_substitutions(self, expression):
        if type(expression) is np.ndarray:
            result = self.numpy_substitutor(expression)
            try:
                return result.astype(float)
            except TypeError:
                return result
        else:
            return expression.subs(self.constants)


class CartPole(PhysicsSystem):
    def __init__(self, M=1.0, m=0.1, L=1.0, b=1.0, g=9.81, operating_point=None,
                 C=None, Q=None, R=None, V=None, W=None):
        constants = dict(zip(sp.symbols('M, m, L, b, g'), [M, m, L, b, g]))
        super().__init__(constants, operating_point, C, Q, R, V, W)

    def create_x_and_u(self):
        x = sp.Function('x')(self.t)
        theta = sp.Function('theta')(self.t)
        f = sp.Function('f')(self.t)
        return np.array([x, theta]), np.array([f])

    def create_lagrangian(self):
        M, m, L, _, g = self.constants.keys()
        x, theta = self.x

        x_rel = -L * sp.sin(theta)
        v_x_rel = sp.diff(x_rel, self.t)
        v_x_pole = sp.diff(x, self.t) + v_x_rel

        y_rel = L * sp.cos(theta)
        v_y_pole = sp.diff(y_rel, self.t)  # relative and absolute velocities are the same

        KE_cart = (M * sp.diff(x, self.t) ** 2) / 2
        KE_pole = (m * v_x_pole ** 2) / 2 + (m * v_y_pole ** 2) / 2
        KE = KE_cart + KE_pole
        PE = m * g * L * sp.cos(theta)
        L = sp.simplify(KE - PE)
        return L

    def create_f_external(self):
        _, _, _, b, _ = self.constants.keys()
        x = self.x[0]
        f = self.u[0]
        f_x_external = f - b * sp.diff(x, self.t)  # dissipative and external forces for the x direction
        f_theta_external = 0
        return np.array([f_x_external, f_theta_external])


class Controller:
    def __init__(self, physics_system):
        self.phys = physics_system

        # lazily initialized
        self.equations_of_motion = None
        self.A = None
        self.B = None
        self.transfer_functions = None
        self.k_pd_mat = None
        self.control_tfs = None

        # lazily initialized, numerical
        self.K = None
        self.L = None
        self.equation_of_motion_funcs = None

    def get_equations_of_motion(self):
        """

        """
        if self.equations_of_motion is not None:
            return self.equations_of_motion

        lagrange_equations = [sp.Eq(sp.diff(sp.diff(self.phys.lagrangian, sp.diff(x, self.phys.t)), self.phys.t),
                                    sp.diff(self.phys.lagrangian, x) + F_external)
                              for x, F_external in zip(self.phys.x, self.phys.f_external)]
        result = sp.solve(lagrange_equations, [sp.diff(x, (self.phys.t, 2)) for x in self.phys.x])

        self.equations_of_motion = np.array([sp.simplify(result[sp.diff(x, (self.phys.t, 2))])
                                             for x in self.phys.x])
        return self.equations_of_motion

    def get_linearized_equations_of_motion(self):
        if self.A is not None and self.B is not None:
            return self.A, self.B

        if self.equations_of_motion is None:
            self.get_equations_of_motion()

        # The operating points are not included in the linearization which means the resulting linear system
        # (x_dot = A * x + B * u) is in reference to the primed coordinates.
        # Thus the K gains for the controller will also correspond to the primed coordinates.
        # However, K * (x_r' - x') = K * (x_r - x) so the controller works the same regardless.
        A_x_dot = np.array([[sp.diff(eq, x).subs(self.phys.operating_point) for x in self.phys.x]
                        for eq in self.equations_of_motion])
        A_x_ddot = np.array([[sp.diff(eq, sp.diff(x, self.phys.t)).subs(self.phys.operating_point) for x in self.phys.x]
                            for eq in self.equations_of_motion])
        A_rows = self.interweave(np.zeros((self.phys.x_dim, self.phys.x_dim)), np.identity(self.phys.x_dim), rows=False)
        self.A = self.interweave(A_rows, self.interweave(A_x_dot, A_x_ddot, rows=False), rows=True)

        B_x_ddot = np.array([[sp.diff(eq, u).subs(self.phys.operating_point) for u in self.phys.u]
                            for eq in self.equations_of_motion])
        B_x_dot = np.zeros_like(B_x_ddot)
        self.B = self.interweave(B_x_dot, B_x_ddot)
        return self.A, self.B

    def get_transfer_functions(self):
        if self.transfer_functions is not None:
            return self.transfer_functions

        if self.A is None or self.B is None:
            self.get_linearized_equations_of_motion()

        laplace_tf = {}
        for x, X in zip(self.phys.x, self.phys.X):
            laplace_tf[x] = X
            laplace_tf[sp.diff(x)] = self.phys.s * X
            laplace_tf[sp.diff(x, (self.phys.t, 2))] = self.phys.s ** 2 * X
        for u, U in zip(self.phys.u, self.phys.U):
            laplace_tf[u] = U

        x_vec = self.interweave(self.phys.x, [sp.diff(x, self.phys.t) for x in self.phys.x])
        linear_eq = np.dot(self.A, x_vec) + np.dot(self.B, self.phys.u)
        laplace_equations = [sp.Eq(sp.diff(x, (self.phys.t, 2)), eq).subs(laplace_tf)
                             for x, eq in zip(self.phys.x, linear_eq[1::2])]
        # need to convert to list because sympy doesn't support numpy arrays
        result = sp.solve(laplace_equations, list(self.phys.X))

        self.transfer_functions = np.array([sp.simplify(result[X]) for X in self.phys.X])
        return self.transfer_functions

    def get_controller_transfer_functions(self):
        if self.control_tfs is not None:
            return self.control_tfs

        if self.transfer_functions is None:
            self.get_transfer_functions()

        k_p_mat = np.array([[sp.symbols(f'k_p_{U.name.lower()}_{X.name.lower()}') for X in self.phys.X]
                            for U in self.phys.U])
        k_d_mat = np.array([[sp.symbols(f'k_d_{U.name.lower()}_{X.name.lower()}') for X in self.phys.X]
                            for U in self.phys.U])

        self.k_pd_mat = np.zeros((self.phys.u_dim, 2 * self.phys.x_dim)).astype(object)
        self.k_pd_mat[:, 0::2] = k_p_mat
        self.k_pd_mat[:, 1::2] = k_d_mat

        U_control = np.dot(k_p_mat, (self.phys.X_r - self.phys.X)) - \
                    np.dot(k_d_mat, self.phys.s * self.phys.X)

        eqs = [sp.Eq(X, tf).subs(zip(self.phys.U, U_control))
               for X, tf in zip(self.phys.X, self.transfer_functions)]
        result = sp.solve(eqs, list(self.phys.X))  # convert to list because sympy doesn't support numpy arrays
        self.control_tfs = np.array([result[X] for X in self.phys.X])

        return self.control_tfs

    def get_lqr_gain(self):
        if self.K is not None:
            return self.K

        if self.A is None or self.B is None:
            self.get_linearized_equations_of_motion()

        self.K = self.linear_quadratic_regulator(self.phys.apply_substitutions(self.A),
                                                 self.phys.apply_substitutions(self.B),
                                                 self.phys.Q, self.phys.R)
        return self.K

    def get_lqe_gain(self):
        if self.L is not None:
            return self.L

        if self.A is None:
            self.get_linearized_equations_of_motion()

        self.L = self.linear_quadratic_regulator(self.phys.apply_substitutions(self.A.T),
                                                 self.phys.apply_substitutions(self.phys.C.T),
                                                 self.phys.V, self.phys.W).T
        return self.L

    def equations_of_motion_func(self, x, u):
        if self.equation_of_motion_funcs is None:
            if self.equations_of_motion is None:
                self.get_equations_of_motion()
            x_params = self.interweave(self.phys.x, [sp.diff(x, self.phys.t) for x in self.phys.x])
            self.equation_of_motion_funcs = [sp.lambdify([x_params, self.phys.u], self.phys.apply_substitutions(eq))
                                             for eq in self.equations_of_motion]

        return self.interweave(x[1::2], np.array([eq(x, u) for eq in self.equation_of_motion_funcs]))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def interweave(x, x_dot, rows=True):
        if len(x.shape) == 1:
            return np.column_stack((x, x_dot)).flatten()
        elif len(x.shape) == 2:
            dtype = float if type(x) == float and type(x_dot) == float else object
            if rows:
                result = np.zeros((2 * x.shape[0], x.shape[1]), dtype)
                result[0::2, :] = x
                result[1::2, :] = x_dot
            else:
                result = np.zeros((x.shape[0], 2 * x.shape[1]), dtype)
                result[:, 0::2] = x
                result[:, 1::2] = x_dot
            return result

    def analyze_controller(self):
        if self.control_tfs is None:
            self.get_controller_transfer_functions()

        for X_r in self.phys.X_r:
            # set all others in X_r to zero
            tfs = [sp.simplify(tf.subs([(X_r_, 0) for X_r_ in self.phys.X_r if X_r_ is not X_r]) / X_r)
                   for tf in self.control_tfs]
            for X, tf in zip(self.phys.X, tfs):
                numerator, denominator = tf.as_numer_denom()

                routh_table = np.array(routh(sp.Poly(denominator, self.phys.s)))
                routh_conditions = routh_table[:, 0]

                tf = self.phys.apply_substitutions(tf)
                numerator = self.phys.apply_substitutions(numerator)
                denominator = self.phys.apply_substitutions(denominator)

                zeros = sp.roots(numerator.subs(zip(self.k_pd_mat.flatten(), self.K.flatten())), self.phys.s)
                poles = sp.roots(denominator.subs(zip(self.k_pd_mat.flatten(), self.K.flatten())), self.phys.s)
                gain = self.get_gain(self.phys.s, tf.subs(zip(self.k_pd_mat.flatten(), self.K.flatten())), zeros, poles)

                zeros_description = ', '.join(
                    [str(zero.evalf(6)) if multiplicity == 1 else f'{zero.evalf(6)} (x{multiplicity})'
                     for zero, multiplicity in zeros.items()])
                poles_description = ', '.join(
                    [str(pole.evalf(6)) if multiplicity == 1 else f'{pole.evalf(6)} (x{multiplicity})'
                     for pole, multiplicity in poles.items()])
                routh_conditions_description = '\n'.join(
                    [to_string(condition.evalf(3)) for condition in routh_conditions])
                print(f'{X}/{X_r}: Routh Conditions:\n{routh_conditions_description}\nGain: {gain.evalf(6)}\n'
                      f'Poles: {poles_description}\nZeros: {zeros_description}\n\n')

    def test_controller(self, x_r_func=None, state_0=None, t_f=10):
        """
        A, B, K, L, and x_0 must be numpy arrays and x_r_func must be a function that returns a numpy array.
        """
        if self.K is None:
            self.get_lqr_gain()
        if self.L is None:
            self.get_lqe_gain()

        if state_0 is None:
            state_0 = np.zeros(self.phys.x_dim + self.phys.y_dim)
        if x_r_func is None:
            # use step input for first variable in x_r
            target = np.zeros(self.phys.x_dim)
            target[0] = 1

            def x_r_func(_):
                return target

        A_num = self.phys.apply_substitutions(self.A)
        B_num = self.phys.apply_substitutions(self.B)

        def state_derivative(t, state):
            print(t)
            mid_point = len(state) // 2
            x = state[:mid_point]
            x_hat = state[mid_point:]

            u = np.dot(self.K, x_r_func(t) - x_hat)
            x_dot = self.equations_of_motion_func(x, u)

            y = np.dot(self.phys.C, x)
            e = y - np.dot(self.phys.C, x_hat)
            print(e)
            x_hat_dot = np.dot(A_num, x_hat) + np.dot(B_num, u) + np.dot(self.L, e)

            return np.concatenate((x_dot, x_hat_dot))

        result = solve_ivp(state_derivative, (0, t_f), state_0, method='RK45', rtol=1e-3)  # , atol=1, max_step=0.01)
        # noinspection PyUnresolvedReferences
        return result.t, result.y


def to_string(expression, to_word=True):
    text = vlatex(expression).replace(r'\operatorname{Theta}', r'\Theta')
    if to_word:
        text = text.replace(r'p f x', r'p,x') \
            .replace(r'd f x', r'd,x') \
            .replace(r'p f \theta', r'p,\theta') \
            .replace(r'd f \theta', r'd,\theta')
    else:
        text = text.replace(r'p f x', r'px') \
            .replace(r'd f x', r'dx') \
            .replace(r'p f \theta', r'pt') \
            .replace(r'd f \theta', r'dt')
    return text


def test_system(physics_system, x_r_func=None, state_0=None, t_f=10):
    """
    Generates

    :param constants: A dictionary that maps sympy variables to floating point values.
                      The sympy variables must be in the order that physics_func expects.
    :param physics_func:
    :param operating_point:
    :param Q:
    :param R:
    :param V:
    :param R:
    :param W:
    :param x_r_func:
    :param x_0:
    :param t_f:
    """
    controller = Controller(physics_system)

    print('For displaying LaTeX:', 'https://latex.codecogs.com/eqneditor/editor.php')

    print('Lagrangian:', to_string(physics_system.lagrangian))
    print('\nExternal Forces:', to_string(physics_system.f_external))

    print('\nEquations of Motion:')
    for x, eq in zip(physics_system.x, controller.get_equations_of_motion()):
        print(to_string(sp.Eq(sp.diff(x, physics_system.t), eq)))

    A, B = controller.get_linearized_equations_of_motion()
    print('\nA (in Primed Variables):\n', to_string(A), '\nB (in Primed Variables):\n', to_string(B))

    transfer_functions = [sp.expand(tf) for tf in controller.get_transfer_functions()]
    print('\nTransfer Functions (in Primed Variables):')
    for X, tf in zip(physics_system.X, transfer_functions):
        print(to_string(sp.Eq(X, tf)))

    print('\nControl Transfer Functions (in Primed Variables):')
    for X, tf in zip(physics_system.X, controller.get_controller_transfer_functions()):
        print(to_string(sp.Eq(X, tf)))

    print('K:\n', controller.get_lqr_gain())
    print('L:\n', controller.get_lqe_gain())

    controller.analyze_controller()

    t_vals, state_vals = controller.test_controller(x_r_func, state_0, t_f)
    forces = [np.dot(controller.K, x_r_func(t_) - state_vals[2 * physics_system.x_dim:, i])[0]
              for i, t_ in enumerate(t_vals)]
    powers = [force * velocity for force, velocity in zip(forces, state_vals[1, :])]
    print('\nMax Force:', np.max(np.abs(forces)))
    print('Max Power Draw:', np.max(np.abs(powers)))
    print('Mean Power Draw:', np.mean(np.abs(powers)))

    return t_vals, state_vals


def main():
    # operating_point = {x_vec[2]: sp.pi}
    cart_pole = CartPole(M=1.994376, m=0.105425, L=0.110996, b=1.6359, g=9.81,
                              C=np.identity(4),
                              operating_point=None, Q=np.diag([10, 20, 100, 50]), R=np.array([[1]]))
    t_vals, state_vals = test_system(cart_pole, x_r_func=lambda t: [0.1, 0, 0, 0],
                                     state_0=np.array([0, 0, 0, 0,
                                                       0, 0, 0, 0]))
    for i, x in enumerate(cart_pole.x):
        plt.plot(t_vals, state_vals[2 * i, :], label=f'${to_string(x)}$')
    for i, x in enumerate(cart_pole.x):
        plt.plot(t_vals, state_vals[2 * (cart_pole.x_dim + i), :], label=f'$\\hat{{{to_string(x)}}}$')

    # plt.plot(t_vals, x_r_func(t_vals)[0], label=r'$x_{r}$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Position (m), Velocity (m/s), $\theta$ (rad), $\omega$ (rad/s)')
    plt.title(r'Recovery from x Perturbation of 15 Meters')
    plt.legend()
    # plt.savefig('theta_perturbation.png')
    plt.show()


if __name__ == '__main__':
    main()
