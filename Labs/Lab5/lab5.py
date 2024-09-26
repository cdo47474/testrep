import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# MODIFIED BISECTION!!!!
# NOT REAL BISECTION CODE!!!!

def driver():
    # use routines
    # pow(x,9) - 45*pow(x,8) + 900*pow(x,7) - 10500*pow(x,6) + 78750*pow(x,5) - 393750*pow(x,4) + 1312500*pow(x,3) - 2812500*pow(x,2) + 3515625*x - 1953125
    f = lambda x: np.exp(np.pow(x,2) + 7*x - 30) - 1
    fprime = lambda x: (2*x + 7)*np.exp(np.pow(x,2) + 7*x - 30)
    a = 2
    b = 4.5
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 1e-10
    Nmax = 100
    [astar,ier] = bisection(f,a,b,tol,fprime)
    print('Midpoint which puts derivative in basin of convergence:',astar)
    print('the error message from bisection reads:',ier)
    print("JUST RETURNED ", astar)
    (p,pstar,info,it) = newton(f,fprime,astar,tol, Nmax)
    print('the approximate root is ', pstar)
    print('the error message from newton reads: ', info)
    print('number of newton iterations:', it)
    print('f(astar) =', f(pstar))
    # define routines
def bisection(f,a,b,tol,fprime):
# Inputs:
# f,a,b - function and endpoints of initial interval
# tol - bisection stops when interval length < tol
# Returns:
# astar - approximation of root
# ier - error message
# - ier = 1 => Failed
# - ier = 0 == success
# first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]
        print("1")
    # verify end points are not a root
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]
    if (fb ==0):
        astar = b
        ier = 0
        return [astar, ier]
    count = 0
    d = 0.5*(a+b)
    print(fprime(d))
    while (abs(fprime(d)) >= 1):
        
        fd = f(d)
        
        if (fd ==0):
            
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd<0):
            
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count = count +1
        print(fprime(d))
    # print('abs(d-a) = ', abs(d-a))
    print("Final slope:", fprime(d))
    print("count: ", count)
    astar = d
    ier = 0
    return [f(d), ier]

def newton(f,fp,p0,tol,Nmax):

    p = np.zeros(Nmax+1)
    print("---------------")
    print(p0)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0- f(p0)/fp(p0)
        
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    print("skip")
    pstar = p1
    info = 1
    
    return [p,pstar,info,it]

driver()
