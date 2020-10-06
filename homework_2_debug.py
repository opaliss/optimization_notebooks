import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random


def rosenbrock_fun(x):
    x1, x2 = x
    """ This function returns the output of the Rosenbrock function."""
    return 100 * ((x2 - x1 ** 2) ** 2) + (1 - x1) ** 2


def rosenbrock_gradient(x):
    x1, x2 = x
    """ return [df/dx1 df/dx2]"""
    dfx1 = -400 * x2 * x1 + 400 * (x1 ** 3) - 2 + 2 * x1
    dfx2 = 200 * x2 - 200 * (x1 ** 2)
    return np.array([dfx1, dfx2])


def rosenbrock_hessian(x):
    x1, x2 = x
    """ return [d2f/dx1^2   d2f/dx1dx2
                d2f/dx1dx2  d2f/dx2^2]"""
    h = np.zeros((2, 2))
    h[0, 0] = -400 * x2 + 1200 * (x1 ** 2) + 2
    h[0, 1] = -400 * x1
    h[1, 0] = -400 * x1
    h[1, 1] = 200
    return h


def pk_steepest_descent(gradient):
    """ search direction for steepest decent."""
    return np.array(-1 * gradient / np.linalg.norm(gradient))


def pk_newton(gradient, hessian):
    """ search direction for Newton's method."""
    h_inv = np.linalg.inv(hessian)
    return -np.matmul(h_inv, gradient)


def phi_function(alpha, pk, xk):
    """ phi(alpha) = f(xk + alpha*pk)"""
    x = xk + alpha * pk
    return rosenbrock_fun(x)


def phi_prime(pk, xk):
    return np.dot(rosenbrock_gradient(xk), pk)


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
        alpha_1 = min(1.0, 1.01 * 2 * (rosenbrock_fun(xk) - rosenbrock_fun(old_x)) / dphi0)
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
            return zoom(alpha_low=alpha_vec[i - 1], alpha_high=alpha_vec[i], xk=xk, pk=pk, c1=c1, c2=c2)

        # compute phi prime at alpha i (ai).
        phi_prime_alpha_i = phi_prime(pk, xk + alpha_i * pk)
        # curvature condition.
        if abs(phi_prime_alpha_i) <= -c2 * dphi0:
            return alpha_i

        if phi_prime_alpha_i >= 0:
            return zoom(alpha_low=alpha_i, alpha_high=alpha_vec[i - 1], xk=xk, pk=pk, c1=c1, c2=c2)

        alpha_vec.append(random.uniform(alpha_i, alpha_max))
        i += 1


def find_local_minimum(x0, c1, c2, alpha, p, tol=1e-8, print_num=None, method="sd", save_xk=True):
    """ Find the local minimum point x* using backtracking line search that will satisfy Armijo-Goldstein inequality.
    The avilable methods: Newton and Steepest Descent. Default is Steepest descent.
    x0 - initial guess for x*.
    c1 - the slope of Armijo-Goldstein line.
    alpha - initial step size.
    p - modify alpha scaler.
    tol - tolerence. the iterative method will stop when ||gradient|| < tol"""

    xk = x0
    k = 0  # iteration number
    alpha_original = alpha

    if save_xk:
        xk_arr = np.array([xk])

    while rosenbrock_fun(xk) > tol and np.linalg.norm(rosenbrock_gradient(xk)) > tol:
        """ find the next iteration xk+1"""
        gradient = rosenbrock_gradient(xk)

        if method == "sd":
            pk = pk_steepest_descent(gradient)

        if method == "newton":
            hessian = rosenbrock_hessian(xk)
            pk = pk_newton(gradient, hessian)

        if print_num is not None:
            if 0 <= k <= 6:
                if k == 0:
                    print("***The first 6 iterations:*** \n")
                print("Iteration #" + str(k) + ", x" + str(k) + " = " + str(xk))
                print("||gradient|| = " + str(np.linalg.norm(gradient)))
                print("f = " + str(rosenbrock_fun(xk)) + "\n")

            if print_num - 5 <= k <= print_num and k > 6:
                if k == print_num - 5 or k == 7:
                    print("***The last 6 iterations:*** \n")
                print("Iteration #" + str(k) + ", x" + str(k) + " = " + str(xk))
                print("||gradient|| = " + str(np.linalg.norm(gradient)))
                print("f = " + str(rosenbrock_fun(xk)) + "\n")

        xk_next = xk + alpha * pk

        while rosenbrock_fun(xk_next) > rosenbrock_fun(xk) + c1 * alpha * np.matmul(pk.T, gradient):
            """ find a step size that will satisfy Armijo-Goldstein inequality. Modify alpha. """
            # print("call line search")
            if k > 1:
                old_x = xk_arr[-4:-2]
            else:
                old_x = None
            alpha = my_line_search(c1=c1, c2=c2, pk=pk, xk=xk, old_x=old_x, alpha_0=0, alpha_max=1, method=method)
            xk_next = xk + alpha * pk

        xk = xk_next
        alpha = alpha_original
        k = k + 1

        if save_xk:
            xk_arr = np.append(xk_arr, [xk])

    print("Iteration #" + str(k) + ", x" + str(k) + " = " + str(xk))
    print("||gradient|| = " + str(np.linalg.norm(rosenbrock_gradient(xk))))
    print("f = " + str(rosenbrock_fun(xk)) + "\n")

    if save_xk:
        return xk, k, xk_arr

    return xk, k


if __name__ == "__main__":
    res_2_sd = find_local_minimum(x0=[1.2, 1.2], c1=1e-4, c2=0.9, alpha=1, p=0.5, tol=1e-8, print_num=23, method="sd",
                                  save_xk=True)

    f_res_2_sd = np.ones(int(len(res_2_sd[-1]) / 2))

    for ii in range(0, int(len(res_2_sd[-1]) / 2)):
        f_res_2_sd[ii] = rosenbrock_fun([res_2_sd[-1][2 * ii], res_2_sd[-1][2 * ii + 1]])

    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(len(f_res_2_sd)), np.log10(f_res_2_sd), 2)

    ax.set_title("Rosenbrock function, ic = [-1.2, 1], SD.")
    ax.set_xlabel("# of iterations")
    ax.set_ylabel("log10(f)")
    plt.show()
