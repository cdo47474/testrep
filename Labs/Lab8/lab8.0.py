import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt

# Need a subroutine that evaulates a line that goes through the points (x0, f(x0)) and (x1, f(x1)) 'at a point alpha' ???? What does 'at a point alpha' mean???

def driver():

    print("Hello")
    f = lambda x: np.pow(np.e, x)
    a = 0
    b = 1

    # Need intervals
    Nint = 10

    fl = line(f, a, b, Nint)
    xeval = np.linspace(-2,2,100)
    fle = fl(xeval)
    f1 = f(xeval)
    plt.plot(xeval, f1)
    plt.plot(xeval, fle)
    plt.show()
    

def line(f, a, b, Nint):

    fa = f(a)
    fb = f(b)
    m = (fb-fa)/(b-a)
    print(m)
    fl = lambda x: m*(x-a) + fa

    xind = np.linspace(a,b)

    return fl


driver()