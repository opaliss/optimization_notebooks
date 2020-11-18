import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rosenbock2Nd import rosenbock2Nd
import random


def phi_function(alpha, pk, xk):
    """ phi(alpha) = f(xk + alpha*pk)"""
    x = xk + alpha * pk
    return rosenbock2Nd(x, 0)


def phi_prime(pk, xk):
    return rosenbock2Nd(xk, 1) @ pk


def hermite(alpha_0, alpha_1, pk, xk):
    """interpolate phi(a0), phi'(a0), phi(a1), phi'(a1)"""
    d1 = phi_prime(pk, xk + alpha_0 * pk) + phi_prime(pk, xk + alpha_1 * pk) - 3 * \
         (phi_function(alpha_0, pk, xk) - phi_function(alpha_1, pk, xk)) / (alpha_0 - alpha_1)
    d2 = np.sign(alpha_1 - alpha_0) * np.sqrt(
        d1 ** 2 - phi_prime(pk, xk + alpha_0 * pk) * phi_prime(pk, xk + alpha_1 * pk))

    frac = (phi_prime(pk, xk + alpha_1 * pk) + d2 - d1) / \
           (phi_prime(pk, xk + alpha_1 * pk) - phi_prime(pk, xk + alpha_0 * pk) + 2 * d2)

    return alpha_1 - (alpha_1 - alpha_0) * frac


def quadradic_interp(alpha_0, pk, xk):
    """ interpolate over phi(0), phi'(0), phi(alpha_0)"""
    top = (alpha_0 ** 2) * (phi_prime(pk, xk))
    bottom = (phi_function(alpha_0, pk, xk) - phi_function(0, pk, xk) - alpha_0 * phi_prime(pk, xk))
    return - top / (2 * bottom)


def cubic_interp(alpha_0, alpha_1, xk, pk):
    # interpolate to the 3rd order
    # over the points: phi(0), phi'(0), phi(alpha_0), phi(alpha_1)
    # the cubic function is in this form:
    # phi_c(alpha) = a*alpha^3 + b* alpha^2 + alpha*phi'(0)  + phi(0)
    coeff = 1 / ((alpha_0 ** 2) * (alpha_1 ** 2) * (alpha_1 - alpha_0))

    mat_1 = np.zeros((2, 2))
    mat_1[0, 0] = alpha_0 ** 2
    mat_1[0, 1] = -alpha_1 ** 2
    mat_1[1, 0] = -alpha_0 ** 3
    mat_1[1, 1] = -alpha_1 ** 3

    mat_2 = np.zeros(2)
    mat_2[0] = phi_function(alpha_1, pk, xk) - phi_function(0, pk, xk) - alpha_1 * phi_prime(pk, xk)
    mat_2[1] = phi_function(alpha_0, pk, xk) - phi_function(0, pk, xk) - alpha_0 * phi_prime(pk, xk)
    ab_vec = coeff * np.matmul(mat_1, mat_2)

    a = ab_vec[0]
    b = ab_vec[1]

    return (-b + np.sqrt(b ** 2 - 3 * a * phi_prime(pk, xk))) / (3 * a)


def interpolation(alpha_0, alpha_1, xk, pk):
    try:
        alpha_star = hermite(alpha_0, alpha_1, pk, xk)
    except:
        return None
    if alpha_star <= 0:
        return None
    # if phi_function(alpha_star, pk, xk) > phi_function(alpha_1, pk, xk) or \
    #         phi_function(alpha_star, pk, xk) > phi_function(alpha_0, pk, xk):
    #     return None --> accounting for concave poly.
    # alpha_range = np.linspace(alpha_0, alpha_1, 25)
    # phi_vals = np.zeros(25)
    # for ii in range(25):
    #     phi_vals[ii] = phi_function(alpha_range[ii], pk, xk)
    # plt.plot(alpha_range, phi_vals)
    # plt.scatter(alpha_star, phi_function(alpha_star, pk, xk))
    # plt.show()
    return hermite(alpha_0, alpha_1, pk, xk)


def zoom(alpha_low, alpha_high, xk, pk, c1, c2):
    """ find xj in the interval of alpha_low and alpha_high. """
    max_iter = 10
    k = 0
    while max_iter > k:
        if abs(alpha_low - alpha_high) < 1e-8:  # safeguard.
            return None
        if phi_function(alpha_high, pk, xk) < phi_function(alpha_low, pk, xk):
            return None

        # interpolate to find xj between alpha_low and alpha_high
        alpha_j = interpolation(alpha_0=alpha_low, alpha_1=alpha_high, xk=xk, pk=pk)

        # if interpolation fails:
        if alpha_j is None:
            alpha_j = (alpha_high - alpha_low) / 2

        # compute phi(xj)
        res = phi_function(alpha_j, pk, xk)
        # test the Armijo condition.
        if (res > phi_function(0, pk, xk) + c1 * alpha_j * phi_prime(pk, xk)) or (
                res >= phi_function(alpha_low, pk, xk)):
            alpha_high = alpha_j

        else:
            # compute phi_prime(x_j)
            if np.abs(phi_prime(pk, xk + alpha_j * pk)) <= -c2 * phi_prime(pk, xk):
                # satisfy the curvature condition.
                return alpha_j

            if phi_prime(pk, xk + alpha_j * pk) * (alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha_j

        k += 1
    return None


def my_line_search(c1, c2, pk, xk, old_x=None, alpha_0=0, alpha_max=1, method="sd"):
    """Find alpha that satisfies strong Wolfe conditions."""
    phi0 = phi_function(0, pk, xk)
    dphi0 = phi_prime(pk, xk)

    # choose alpha_1
    if old_x is not None and dphi0 != 0 and method == "sd":
        alpha_1 = min(1.0, 1.01 * 2 * (rosenbock2Nd(xk, 0) - rosenbock2Nd(old_x, 0)) / dphi0)
    else:
        alpha_1 = 1.0

    if alpha_1 <= 0:
        alpha_1 = 1.0

    if alpha_max is not None:
        alpha_1 = min(alpha_1, alpha_max)

    alpha_vec = [alpha_0, alpha_1]

    i = 1
    while True:
        # alpha i = ai
        alpha_i = alpha_vec[i]
        # compute phi(ai)
        phi_i = phi_function(alpha_i, pk, xk)
        # Armijo condition.
        if phi_i > phi0 + c1 * alpha_i * dphi0 \
                or (i > 1 and phi_function(alpha_i, pk, xk) >= phi_function(alpha_vec[i - 1], pk, xk)):
            return zoom(alpha_low=alpha_vec[i - 1], alpha_high=alpha_vec[i], xk=xk, pk=pk, c1=c1, c2=c2), i

        # compute phi prime at alpha i (ai).
        phi_prime_alpha_i = phi_prime(pk, xk + alpha_i * pk)
        # curvature condition.
        if abs(phi_prime_alpha_i) <= -c2 * dphi0:
            return alpha_i, i

        if phi_prime_alpha_i >= 0:
            return zoom(alpha_low=alpha_i, alpha_high=alpha_vec[i - 1], xk=xk, pk=pk, c1=c1, c2=c2), i

        alpha_vec.append(random.uniform(alpha_i, alpha_max))
        i += 1

def rosen(x):
    """Generalized n-dimensional version of the Rosenbrock function"""
    return sum(100*(x[1:]-x[:-1]**2.0)**2.0 +(1-x[:-1])**2.0)

def rosen_der(x):
    """Derivative of generalized Rosen function."""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def bfgs_method(x0, eps=1e-6, H0=np.eye(18),c1=1e-4):
    """ x0 - initial starting point (dim2)
        eps - default is 1e-8
        H0 - default is the identity matrix.
    """
    k = 0  # initialize num of outer iterations.
    inner_k = 0  # initialize inner k iteration.
    old_xk = None
    alpha_original = 1
    alpha = np.copy(alpha_original)
    xk = x0  # intitialize x.
    Hk = H0  # initialize H, positive definite matrix.
    I = np.eye(len(x0))  # idenitity matrix of 2 by 2.

    alpha_vec = []
    f_vec = []
    grad_vec = []
    inner_k = []
    conv_c = []

    while np.linalg.norm(rosen_der(xk)) > eps:
        pk = -Hk @ rosen_der(xk)

        xk_next = xk + alpha * pk
        ink = 0
        print(xk)
        while rosen(xk_next) > rosen(xk) + c1 * alpha * (pk.T @ rosen_der(xk)):
            """ find a step size that will satisfy Armijo-Goldstein inequality. Modify alpha. """
            alpha = 0.1* alpha
            xk_next = xk + alpha * pk
            ink += 1

        inner_k.append(abs(int(ink)))

        xk_next = xk + alpha * pk

        sk = xk_next - xk

        yk = rosen_der(xk_next) - rosen_der(xk)

        rho = 1 / (yk.T @ sk)

        Hk = np.copy((I - rho * sk @ yk.T) @ Hk @ (I - rho * yk @ sk.T) + rho * sk @ sk.T)

        old_xk = np.copy(xk)
        xk = np.copy(xk_next)

        alpha_vec.append(alpha)
        f_vec.append(rosen(xk))
        grad_vec.append(np.linalg.norm(rosen_der(xk)))
        alpha = np.copy(alpha_original)
        print(f_vec[-1])

        k += 1

    return xk, k, inner_k, alpha_vec, f_vec, grad_vec


if __name__ == "__main__":
    x0 = rosenbock2Nd(np.array([1.2, 1.2]), -1)
    print("initial f(x0) = ", rosenbock2Nd(x0, 0))
    xk, k, inner_k, alpha_vec, f_vec, grad_vec = bfgs_method(x0)