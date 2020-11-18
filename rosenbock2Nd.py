import numpy as np

def rosenbock2Nd(x,order):
    if (order == -1):
        xN = np.array([ 1.0 , 1.0 ])
        x0easy = np.array([  1.2 , 1.2 ])
        x0e2   = (xN + x0easy) / 2
        x0e3   = (xN + x0e2)  / 2
        x0e4   = (xN + x0e3) / 2
        x0hard = np.array([ -1.2 , 1.0 ])
        x0h2   = (xN + x0hard) / 2
        x0h3   = (xN + x0h2)  / 2
        x0h4   = (xN + x0h3) / 2
        x0h5   = 2*x0hard
        R      = np.concatenate((x0easy, x0e2, x0e3, x0e4, x0hard, x0h2, x0h3, x0h4, x0h5))
        return R

    rb2d      = lambda x: ( 100.0*(x[1]-x[0]**2)**2 + (1-x[0])**2 )
    rb2d_x    = lambda x: ( -400*(x[1]-x[0]**2)*x[0]-2+2*x[0] )
    rb2d_xx   = lambda x: ( 1200*x[0]**2-400*x[1]+2 )
    rb2d_xy   = lambda x: ( -400*x[0] )
    rb2d_y    = lambda x: ( 200*x[1]-200*x[0]**2 )
    rb2d_yy   = lambda x: ( 200 )
    rb2d_grad = lambda x: np.array( [rb2d_x(x),rb2d_y(x)] )
    rb2d_hess = lambda x: np.array( [[rb2d_xx(x), rb2d_xy(x)], [rb2d_xy(x), rb2d_yy(x)]] );

    if (order == 0):
        R = np.zeros(1)
        for k in range(0,len(x),2):
            R = R + rb2d(([x[k],x[k+1]]))
        return R

    elif (order == 1):
        R = np.zeros(len(x))
        for k in range(0,len(x),2):
            val    = rb2d_grad(([x[k],x[k+1]]))
            R[k]   = val[0]
            R[k+1] = val[1]
        return R

    elif (order == 2):
        R = np.zeros((len(x),len(x)))
        for k in range(0,len(x),2):
            val         = rb2d_hess(([x[k],x[k+1]])) #returns 18x18 matrix
            R[k][k]     = val[0][0]
            R[k][k+1]   = val[0][1]
            R[k+1][k]   = val[1][0]
            R[k+1][k+1] = val[1][1]
        return R

    else:
        print("Cannot compute derivatives of order %d", order)

if __name__ == "__main__":
    x0 = rosenbock2Nd(np.array([1.2, 1.2]), -1)
    print(x0)
    fun = rosenbock2Nd(x0, 0)
    print(fun)
    grad = rosenbock2Nd(x0, 1)
    print(grad)
    hes = rosenbock2Nd(x0, 2)
    print(hes)