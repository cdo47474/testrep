import numpy as np
from scipy import special
import matplotlib.pyplot as plt

Ts = -15
Ti = 20
alpha = 0.138e-6
t = 5.184e6
x = np.linspace(0, 10, 100)
# T = lambda x: Ts + (2/(np.sqrt(np.pi)*(Ti -Ts))) * special.erf(x/(2*np.sqrt(alpha*t)))
# Tprime = lambda x: (2/(np.sqrt(np.pi)*(Ti -Ts))) * np.exp(np.pow(-(x/(2*np.sqrt(alpha*t))), 2))
T = Ts + ((2 * (Ti -Ts))/(np.sqrt(np.pi))) * special.erf(0.591*x)
Tprime = ((2 * (Ti -Ts))/(np.sqrt(np.pi))) * np.exp(np.pow((-0.591*x), 2))
# Horrible

# plt.plot(x, T)
# plt.title("Function f on [0,x=10]")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(True)
# plt.show()

# NEWTONS METHOD

def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2
    
    f = lambda x: Ts + ((2 * (Ti -Ts))/(np.sqrt(np.pi))) * special.erf(0.591*x)
    fp = lambda x: ((2 * (Ti - Ts))/(np.sqrt(np.pi))) * np.exp(-1 * np.pow((0.591*x), 2))
    p0 = 0.01
    Nmax = 100
    tol = 1.e-13
    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
def newton(f,fp,p0,tol,Nmax):

    p = np.zeros(Nmax+1)
    print("Here")
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        
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

