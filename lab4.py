import numpy as np
import matplotlib.pyplot as plt


def driver():
    # test functions
    # f1 = lambda x: x * (np.pow((1 + (7 - np.pow(x, 5))/(x * x)), 3)) OVERFLOW
    # f1 = lambda x: x - (np.pow(x, 5) - 7)/(x*x) OVERFLOW
    # f1 = lambda x: x - (np.pow(x, 5) - 7)/(5*np.pow(x, 4)) WORKS!
    f1 = lambda x: np.pow(10/(x+4), 1/2)
    # fixed point is alpha1 = 1.4987....
    # f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09...
    Nmax = 100
    tol = 1e-10
# test f1 '''
    x0 = 1.5
    x = np.zeros((0,1))
    [xstar,ier, x] = fixedpt(f1,x0,tol,Nmax, x)
    fit = compute_order(x[:-1], xstar)
    # pstar = aitkens(x, xstar)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
    print('-----------------------------')
    # print('aitkens number is ', pstar)
#test f2 '''
    # x0 = 0.0
    # [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
    # print('the approximate fixed point is:',xstar)
    # print('f2(xstar):',f2(xstar))
    # print('Error message reads:',ier)
# define routines
def fixedpt(f,x0,tol,Nmax, x):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            x = np.append(x,x1)
            print(count)
            return [xstar,ier, x]
        x = np.append(x,x1)
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier, x]

# def aitkens(x):

    # numerator = np.pow((x[1::] - xstar),2)
    # denominator = (x[2::] - 2*(x[1::]) + xstar)
    # pstar = xstar - (numerator/denominator)

    # xn = x[:-2]
    # xn1 = x[1:-1]
    # xn2 = x[2:]
    # return xn - (np.pow((xn1 - xn),2))/


    # return xn -

# Still working on aitkens

def compute_order(x, xstar):
    

    diff1 = np.abs(x[1::] - xstar)
    diff2 = np.abs(x[0:-1] - xstar)
    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)

    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda is {_lambda}")
    print(f"alpha is {alpha}")

    return fit

driver()

# Exercises:
# 2.2a - It took the code above 12 iterations to reach the desired fixed point to an accuracy of 1e-10
# 2.2b - The order of convergence is 1, so linear convergence! (alpha = 0.996799)



