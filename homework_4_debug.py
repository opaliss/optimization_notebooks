import numpy as np
import matplotlib.pyplot as plt


def rosenbrock_fun(xk):
    """ This function returns the output of the Rosenbrock function."""
    x1, x2 = xk
    return 100 * ((x2 - x1 ** 2) ** 2) + (1 - x1) ** 2


def rosenbrock_gradient(xk):
    """ return [df/dx1 df/dx2]"""
    x1, x2 = xk
    dfx1 = -400 * x2 * x1 + 400 * (x1 ** 3) - 2 + 2 * x1
    dfx2 = 200 * x2 - 200 * (x1 ** 2)
    return np.array([dfx1, dfx2])


def rosenbrock_hessian(xk):
    """ return [d2f/dx1^2   d2f/dx1dx2
                d2f/dx1dx2  d2f/dx2^2]"""
    x1, x2 = xk
    h = np.zeros((2, 2))
    h[0, 0] = -400 * x2 + 1200 * (x1 ** 2) + 2
    h[0, 1] = -400 * x1
    h[1, 0] = -400 * x1
    h[1, 1] = 200
    return h


def mk_fun(xk, pk):
    """ mk taylor approximation of the objective function"""
    Bk = rosenbrock_hessian(xk)
    return rosenbrock_fun(xk) + np.dot(pk, rosenbrock_gradient(xk)) + 0.5 * np.dot(pk, np.matmul(Bk, pk))


def rho_k(xk, pk):
    """ return rho_k = (f(xk) - f(xk+pk))/(mk(0) - mk(pk))"""
    return (rosenbrock_fun(xk) - rosenbrock_fun(xk + pk)) / (mk_fun(xk, [0, 0]) - mk_fun(xk, pk))


def get_pk_fs(gradient, hessian):
    """ search direction for Newton's method."""
    h_inv = np.linalg.inv(hessian)
    return -np.matmul(h_inv, gradient)


def find_tau(pj, dj, delta):
    """ find tau that satisfies ||pj + tau*dj|| = delta"""
    djp = np.dot(dj, pj)
    djd = np.dot(dj, dj)
    pjp = np.dot(pj, pj)
    res1 = (-2 * (djp) + np.sqrt((2 * djp) ** 2 - 4 * djd * (pjp - delta ** 2))) / (2 * djd)
    res2 = (-2 * (djp) - np.sqrt((2 * djp) ** 2 - 4 * djd * (pjp - delta ** 2))) / (2 * djd)
    if res1 >= 0 and res2 < 0:
        return res1
    elif res2 >= 0 and res1 < 0:
        return res2
    elif res1 >= 0 and res2 >= 0:
        return False
    else:
        return False


def steihaug(x0, delta, eps=1e-6):
    pk = np.zeros(2)
    r0 = rosenbrock_gradient(x0)
    rk = np.copy(r0)
    d = -np.copy(r0)
    xk = np.copy(x0)
    Bk = rosenbrock_hessian(xk)


    while True:
        # negative curvature
        if d.T @ Bk @ d <= 0:
            # find tau that satisfies ||pj + tau*dj|| = delta
            print("Steihaug - negative curvature")
            tau = find_tau(pk, d, delta)
            print("tau = ", tau)
            return pk + tau * d

        alpha = np.dot(rk, rk) / (d.T @ Bk @ d)
        p_next = pk + alpha * d

        # step outside trust region.
        if np.linalg.norm(p_next) >= delta:
            # find tau such that ||pj + tau*dj|| = delta
            tau = find_tau(pk, d, delta)
            print("Steihaug - reached the trust-region boundary")
            print("tau = ", tau)
            return pk + tau * d

        r_next = rk + alpha * Bk @ d

        if np.linalg.norm(r_next) <= eps * np.linalg.norm(r0):
            print("Steihaug - met the stopping test.")
            return p_next

        b = np.dot(r_next, r_next) / np.dot(rk, rk)
        d = -r_next + b * d
        rk = np.copy(r_next)
        pk = np.copy(p_next)


def trust_region(x0, delta=1, delta_max=300, eta=1e-3, tol=1e-8):
    k = 0
    xk = x0
    xk_vec = [x0]
    delta_prev = delta

    # while optimality is not satisfied.
    while np.linalg.norm(rosenbrock_gradient(xk)) > tol:
        print("iteration #", k)
        # get pk approximate solution. Using steihaug method.
        pk = steihaug(xk, delta, eps=1e-6)

        # evaluate rho_k
        rk = rho_k(xk, pk)

        if rk < 0.25:
            delta = 0.25 * delta

        else:
            if rk > 0.75 and np.linalg.norm(pk) - delta < 1e-8:
                delta = min(2 * delta, delta_max)

        if rk > eta:
            xk = xk + pk

        else:
            xk = xk

        k += 1
        print("f = ", rosenbrock_fun(xk))
        print("||gradient(f(x))|| = ", np.linalg.norm(rosenbrock_gradient(xk)))
        print("xk = ", xk)
        print("delta = ", delta)
        print("\n")
        xk_vec.append(xk)

    return xk, k, pk, xk_vec


if __name__ == "__main__":
    trust_region([2.8, 4.])
