import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# NEWTONS METHOD

# f = 

def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2
    
    f = lambda x: np.exp(3*x) - 27*np.pow(x,6) + 27*np.pow(x,4)*np.exp(x) - 9*np.pow(x,2)*np.exp(2*x)
    fp = lambda x: 3*np.exp(3*x) - 162*np.pow(x,5) + 108*np.pow(x,3)*np.exp(x) + 27*np.pow(x,4)*np.exp(x) - 18*x*np.exp(2*x) - 18*np.pow(x,2)*np.exp(2*x)
    p0 = 3
    m = 3.7330922341
    Nmax = 100
    tol = 1.e-10
    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax,m)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
def newton(f,fp,p0,tol,Nmax,m):

    p = np.zeros(Nmax+1)
    print("Here")
    print(p0)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0- m*(f(p0)/fp(p0))
        
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    
    return [p,pstar,info,it]
driver()

