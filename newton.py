import numpy as np

def rosenbrock_fun(x1, x2):
    """ This function returns the output of the Rosenbrock function."""
    return 100*((x2 - x1**2)**2) + (1 - x1)**2

def rosenbrock_gradient(x1, x2):
    """ return [df/dx1 df/dx2]"""
    dfx1 = -400*x2*x1 + 400*(x1**3) - 2 + 2*x1
    dfx2 = 200*x2 - 200*(x1**2)
    return np.array([dfx1, dfx2])

def rosenbrock_hessian(x1, x2):
    """ return [d2f/dx1^2   d2f/dx1dx2
                d2f/dx1dx2  d2f/dx2^2]"""
    h = np.zeros((2, 2))
    h[0, 0] = -400*x2 + 1200*(x1**2) + 2
    h[0, 1] = -400*x1
    h[1, 0] = -400*x1
    h[1, 1] = 200
    return h

def pk_steepest_descent(gradient):
    """ search direction for steepest decent."""
    return np.array(-1*gradient/np.linalg.norm(gradient))

def pk_newton(gradient, hessian):
    """ search direction for Newton's method."""
    h_inv = np.linalg.inv(hessian)
    return -np.matmul(h_inv, gradient)


def find_local_minimum(x0, c1, alpha, p, tol=1e-8, print_num=None, method="sd", save_xk=True):
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
        alpha_vec = []
        grad_vec = []
        f_vec = []
        inner_k = []

    while np.linalg.norm(rosenbrock_gradient(xk[0], xk[1])) > tol:
        """ find the next iteration xk+1"""
        gradient = rosenbrock_gradient(xk[0], xk[1])

        if method == "sd":
            pk = pk_steepest_descent(gradient)

        if method == "newton":
            hessian = rosenbrock_hessian(xk[0], xk[1])
            pk = pk_newton(gradient, hessian)

        if print_num is not None:
            if 0 <= k <= 6:
                if k == 0:
                    print("***The first 6 iterations:*** \n")
                print("Iteration #" + str(k) + ", x" + str(k) + " = " + str(xk))
                print("||gradient|| = " + str(np.linalg.norm(gradient)))
                print("f = " + str(rosenbrock_fun(xk[0], xk[1])) + "\n")

            if print_num - 5 <= k <= print_num and k > 6:
                if k == print_num - 5 or k == 7:
                    print("***The last 6 iterations:*** \n")
                print("Iteration #" + str(k) + ", x" + str(k) + " = " + str(xk))
                print("||gradient|| = " + str(np.linalg.norm(gradient)))
                print("f = " + str(rosenbrock_fun(xk[0], xk[1])) + "\n")

        xk_next = xk + alpha * pk
        ii = 1
        while rosenbrock_fun(xk_next[0], xk_next[1]) > rosenbrock_fun(xk[0], xk[1]) + c1 * alpha * np.matmul(pk.T,gradient):
            """ find a step size that will satisfy Armijo-Goldstein inequality. Modify alpha. """
            alpha = p * alpha
            xk_next = xk + alpha * pk
            ii+=1

        alpha_vec.append(alpha)
        f_vec.append(rosenbrock_fun(xk_next[0], xk_next[1]))
        grad_vec.append(np.linalg.norm(rosenbrock_gradient(xk_next[0], xk_next[1])))
        inner_k.append(int(ii))
        xk = xk_next

        alpha = alpha_original
        k = k + 1

        if save_xk:
            xk_arr = np.append(xk_arr, [xk])

    if save_xk:
        return xk, k, inner_k, alpha_vec, f_vec, grad_vec

    print("Iteration #" + str(k) + ", x" + str(k) + " = " + str(xk))
    print("||gradient|| = " + str(np.linalg.norm(rosenbrock_gradient(xk[0], xk[1]))))
    print("f = " + str(rosenbrock_fun(xk[0], xk[1])) + "\n")

    return xk, k