# get lgwts routine and numpy
from scipy.integrate import quad
import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt
# adaptive quad subroutines
# the following three can be passed
# as the method parameter to the main adaptive_quad() function

def driver():
    f = lambda x: np.sin(1/x)
    a = 0.1
    b = 2
    N = 6
    tol = 10e-3
    method = eval_composite_trap

    intreal,err = quad(f,a,b)
    print("The actual integral evaluates to: ", intreal)

    I,X,nsplit = adaptive_quad(a,b,f,tol,N,method)
    print(I,X,nsplit)
    err2 = abs(intreal - I)
    print("The error for this approximation is: ", err2)


def lgwt(N,a,b):
  """ 
   This script is for computing definite integrals using Legendre-Gauss 
   Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
   [a,b] with truncation order N
  
   Suppose you have a continuous function f(x) which is defined on [a,b]
   which you can evaluate at any x in [a,b]. Simply evaluate it at all of
   the values contained in the x vector to obtain a vector f. Then compute
   the definite integral using np.sum(f*w)
  
   Written by Greg von Winckel - 02/25/2004
   translated to Python - 10/30/2022
  """
  N = N-1
  N1 = N+1
  N2 = N+2
  eps = np.finfo(float).eps  
  xu = np.linspace(-1,1,N1)
  
  # Initial guess
  y = np.cos((2*np.arange(0,N1)+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)

  # Legendre-Gauss Vandermonde Matrix
  L = np.zeros((N1,N2))
  
  # Compute the zeros of the N+1 Legendre Polynomial
  # using the recursion relation and the Newton-Raphson method
  
  y0 = 2.
  one = np.ones((N1,))
  zero = np.zeros((N1,))

  # Iterate until new points are uniformly within epsilon of old points
  while np.max(np.abs(y-y0)) > eps:
      
    L[:,0] = one
    
    L[:,1] = y
    for k in range(2,N1+1): 
      L[:,k] = ((2*k-1)*y*L[:,k-1]-(k-1)*L[:,k-2])/k
    
    lp = N2*(L[:,N1-1]-y*L[:,N2-1])/(1-y**2)   
    
    y0 = y
    y = y0-L[:,N2-1]/lp
    
    
  
  # Linear map from[-1,1] to [a,b]
  x=(a*(1-y)+b*(1+y))/2
  
  # Compute the weights
  w=(b-a)/((1-y**2)*lp**2)*(N2/N1)**2
  return x,w


def eval_composite_trap(N,a,b,f):
    """
    put code from prelab with same returns as gauss_quad
    you can return None for the weights
    """
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
    x1 = 0
    _ = 0
    return int_sum,x1,_

def eval_composite_simpsons(N,a,b,f):
    """
    put code from prelab with same returns as gauss_quad
    you can return None for the weights
    """

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

    x1 = 0
    _ = 0
    return int_sum,x1,_


def eval_gauss_quad(M,a,b,f):
    """
    Non-adaptive numerical integrator for \int_a^b f(x)w(x)dx
    Input:
    M - number of quadrature nodes
    a,b - interval [a,b]
    f - function to integrate

    Output:
    I_hat - approx integral
    x - quadrature nodes
    w - quadrature weights

    Currently uses Gauss-Legendre rule
    """
    x,w = lgwt(M,a,b)
    I_hat = np.sum(f(x)*w)
    return I_hat,x,w

def adaptive_quad(a,b,f,tol,M,method):
    """
    Adaptive numerical integrator for \int_a^b f(x)dx

    Input:
    a,b - interval [a,b]
    f - function to integrate
    tol - absolute accuracy goal
    M - number of quadrature nodes per bisected interval
    method - function handle for integrating on subinterval
            - eg) eval_gauss_quad, eval_composite_simpsons etc.

    Output: I - the approximate integral
            X - final adapted grid nodes
            nsplit - number of interval splits
    """
    # 1/2^50 ~ 1e-15
    maxit = 50
    left_p = np.zeros((maxit,))
    right_p = np.zeros((maxit,))
    s = np.zeros((maxit,1))
    left_p[0] = a; right_p[0] = b;
    # initial approx and grid
    s[0],x,_ = method(M,a,b,f);
    # save grid
    X = []
    X.append(x)
    j = 1;
    I = 0;
    nsplit = 1;
    while j < maxit:
    # get midpoint to split interval into left and right
        c = 0.5*(left_p[j-1]+right_p[j-1]);
        # compute integral on left and right spilt intervals
        s1,x,_ = method(M,left_p[j-1],c,f); X.append(x)
        s2,x,_ = method(M,c,right_p[j-1],f); X.append(x)
        if np.max(np.abs(s1+s2-s[j-1])) > tol:
            left_p[j] = left_p[j-1]
            right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
            s[j] = s1
            left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
            s[j-1] = s2
            j = j+1
            nsplit = nsplit+1
        else:
            I = I+s1+s2
            j = j-1
            if j == 0:
                j = maxit
    return I,np.unique(X),nsplit

driver()