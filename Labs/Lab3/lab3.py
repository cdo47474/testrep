import numpy as np
import matplotlib.pyplot as plt

# BISECTION:

def driver():
    # use routines
    # pow(x,9) - 45*pow(x,8) + 900*pow(x,7) - 10500*pow(x,6) + 78750*pow(x,5) - 393750*pow(x,4) + 1312500*pow(x,3) - 2812500*pow(x,2) + 3515625*x - 1953125
    f = lambda x: x
    a = 1
    b = 4
    # f = lambda x: np.sin(x)
    # a = 0.1
    # b = np.pi+0.1
    tol = 1e-10
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    # define routines
def bisection(f,a,b,tol):
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
    while (abs(d-a)> tol):
        
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
    # print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    print(count)
    return [astar, ier]
driver()


# FIXED POINT:


# import libraries

# def driver():
#     # test functions
#     # f1 = lambda x: x * (np.pow((1 + (7 - np.pow(x, 5))/(x * x)), 3)) OVERFLOW
#     # f1 = lambda x: x - (np.pow(x, 5) - 7)/(x*x) OVERFLOW
#     # f1 = lambda x: x - (np.pow(x, 5) - 7)/(5*np.pow(x, 4)) WORKS!
#     f1 = lambda x: x - (np.pow(x, 5) - 7)/(12) 
#     # fixed point is alpha1 = 1.4987....
#     # f2 = lambda x: 3+2*np.sin(x)
# #fixed point is alpha2 = 3.09...
#     Nmax = 100
#     tol = 1e-10
# # test f1 '''
#     x0 = 1
#     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
#     print('the approximate fixed point is:',xstar)
#     print('f1(xstar):',f1(xstar))
#     print('Error message reads:',ier)
# #test f2 '''
#     # x0 = 0.0
#     # [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
#     # print('the approximate fixed point is:',xstar)
#     # print('f2(xstar):',f2(xstar))
#     # print('Error message reads:',ier)
# # define routines
# def fixedpt(f,x0,tol,Nmax):
#     ''' x0 = initial guess'''
#     ''' Nmax = max number of iterations'''
#     ''' tol = stopping tolerance'''
#     count = 0
#     while (count <Nmax):
#         count = count +1
#         x1 = f(x0)
#         if (abs(x1-x0) <tol):
#             xstar = x1
#             ier = 0
#             return [xstar,ier]
#         x0 = x1
#     xstar = x1
#     ier = 1
#     return [xstar, ier]
# driver()



# Lab questions:
# For the first 3 intervals with bisection, the first and third were able to find the root at (1,0), but none of the functions were able
# to find the root at (0,0), since the function never crosses the x axis, so any interval will fail the first check of fa*fb > 0

# 2a. Yes, Yes, Yes
# 2b. We chose an interval around an even root, so bisection did not pick it up and gave an error, the code worked and this is what we expected, the code could not find the correct root
# 2c. Yes, the root was one of the endpoints of the interval, and the code picked it up. And with the changed interval, the code picked up that there was no root in the interval

# 3. The first two functions did not converge, this is because the slope at x0 is larger than one, so the function got farther away from the line y=x with each iteration
# However the other two, the third and fourth ones, did converge.