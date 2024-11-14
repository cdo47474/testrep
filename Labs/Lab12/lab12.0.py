from scipy.integrate import quad
import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt


# Gonna try and make code for composite trapezoidal quadrature, and composite simpsons rule quadrature

def driver():

    f = lambda x: np.sin(1/x)
    a = 2
    b = 0.1
    N = 100
    xeval = np.linspace(a,b,N+1)
    # fex = f(xeval)
    # plt.plot(xeval, fex)
    # plt.show()
    ''' the integral of f(x) over [-1,1] is 2.350402387'''
    intsum = comp_trap(a,b,N,f)

    intsum2 = comp_simp(a,b,N,f)

    intreal,errq = quad(f,a,b)
    print("The actual integral evaluates to: ", intreal)
    print("---------------------------------------------------------")

    print("Integral of f(x) evaluated by Composite Trapezoidal rule is: ", intsum)
    err = intsum - intreal
    print("The error of the Composite Trapezoidal rule is: ", err)

    print("Integral of f(x) evaluated by Composite Simpson rule is: ", intsum2)
    err2 = intsum2 - intreal
    print("The error of the Composite Simpson rule is: ", err2)

    
def comp_trap(a,b,N,f):
    h = (b-a)/N
    xeval = np.linspace(a,b,N+1)
    w = np.zeros(N+1)
    feval = np.zeros(N+1)
    int_sum = 0
    temp = 0
    for kk in range (N+1):
        feval[kk] = f(xeval[kk])
        w[kk] = 2
    w[0] = 1
    w[N] = 1

    for ii in range (N+1):
        temp = w[ii]*feval[ii]
        
        int_sum = int_sum + temp

    int_sum = (h/2)*int_sum

    return int_sum

def comp_simp(a,b,N,f):
    h = (b-a)/N
    xeval = np.linspace(a,b,N+1)
    w = np.zeros(N+1)
    feval = np.zeros(N+1)
    int_sum = 0
    temp = 0
    for kk in range (N+1):
        feval[kk] = f(xeval[kk])
        w[kk] = 2
        if((kk % 2) == 1):
            w[kk] = 4
    w[0] = 1
    w[N] = 1
    
    for ii in range (N+1):
        temp = w[ii]*feval[ii]
        
        int_sum = int_sum + temp

    int_sum = (h/3)*int_sum

    return int_sum

driver()