import numpy as np
import sympy as sp
from tbcontrol.symbolic import routh


def routh_table():
    s = sp.symbols('s')
    K, a, b, c, d = sp.symbols('K, a, b, c, d')
    denominator = s ** 2 + K * s + 2.5 * K - 4
    table = np.array(routh(sp.Poly(denominator, s)))

    print(table)


def hydrofoil():
    # 38.8in*sin(9.5 degrees)*4.76in*19m/s*997kg/m^3*19m/s*tan(9.5 degrees)
    theta = sp.symbols('theta')
    U_inf = sp.symbols('U_inf')
    a = sp.symbols('a')  # 1.12 * scaling_factor
    rho = sp.symbols('rho')
    alpha = sp.symbols('alpha')  # np.radians(9.5)
    u_y = sp.symbols('u_y')  # 0.105 * scaling_factor
    Gamma = 4 * sp.pi * U_inf * a * sp.sin(alpha + sp.asin(u_y / a))
    v_theta = Gamma / (2 * sp.pi * a) + 2 * U_inf * sp.sin(theta)
    p = (rho / 2) * (U_inf ** 2 - v_theta ** 2)
    val = sp.Integral(-p * sp.sin(theta) * a, (theta, 0, 2 * sp.pi))
    print(sp.simplify(val))
    print(val.subs([(U_inf, 19), (a, 1.1291), (rho, 997),
                                 (alpha, np.radians(9.5)), (u_y, 0.07)]).evalf() * 0.120904 / (2.04 + 2.))


if __name__ == '__main__':
    hydrofoil()
