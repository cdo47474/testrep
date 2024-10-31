import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt

def driver():
    a = -1
    b = 1
    xeval = np.linspace(a,b)
    f = lambda x: math.exp(x)
    x1 = 2
    N = 5
    
    p = eval_legendre(x1, N, xeval)
    print(p)



def eval_legendre(x, n, xeval):

    # Given phi0 and phi1, can i find phi2?
    phi = np.zeros(n+1)

    phi[0] = 1
    phi[1] = x

    for j in range (n-1):
        jn = j+2
        phi[j+2] = (2*jn + 1)/(jn + 1) * x * phi[j+1] - jn/(jn + 1) * phi[j]

    return phi


driver()