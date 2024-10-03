import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Ts = -15
# Ti = 20
# alpha = 0.138e-6
# t = 5.184e6
# x = np.linspace(0, 10, 100)
# # T = lambda x: Ts + (2/(np.sqrt(np.pi)*(Ti -Ts))) * special.erf(x/(2*np.sqrt(alpha*t)))
# # Tprime = lambda x: (2/(np.sqrt(np.pi)*(Ti -Ts))) * np.exp(np.pow(-(x/(2*np.sqrt(alpha*t))), 2))
# T = Ts + ((2 * (Ti -Ts))/(np.sqrt(np.pi))) * special.erf(0.591*x)
# Tprime = ((2 * (Ti -Ts))/(np.sqrt(np.pi))) * np.exp(np.pow((-0.591*x), 2))
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
    
    # f = lambda x: Ts + ((2 * (Ti -Ts))/(np.sqrt(np.pi))) * special.erf(0.591*x)
    # fp = lambda x: ((2 * (Ti - Ts) * 0.591)/(np.sqrt(np.pi))) * np.exp(-1 * np.pow((0.591*x), 2))
    f = lambda x: np.pow(x,6) - x - 1
    fp = lambda x: 6*np.pow(x,5) - 1
    p0 = 2 
    x0 = 2
    x1 = 1
    Nmax = 100
    tol = 1.e-13
    x = np.zeros((0,1))
    y = np.zeros((0,1))

    (p,pstar,info,it,x) = newton(f,fp,p0,tol, Nmax,x)
    print(x)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

    (pstar2, ier,y) = secant(x0,x1,f,tol, Nmax,y)
    print(y)
    print('the approximate root is ', pstar2)
    print('the error message reads ', ier)

    # Can we make a graph from the arrays I created?

    count = 0
    while (count < np.size(y) - 1):
        
        
        y1 = np.log10(np.abs(y[count + 1]))
        x1 = np.log10(np.abs(y[count]))
        plt.scatter(x1,y1)
        count = count + 1

    plt.xlabel('log|xn - alpha|')
    plt.ylabel('log|xn+1 - alpha|')
    plt.title('Plot for Secant')
    plt.show()


def newton(f,fp,p0,tol,Nmax,x):

    p = np.zeros(Nmax+1)
    root = 1.134724138015194
    p[0] = p0
    x = np.append(x,(p0 - root))
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            
            pstar = p1
            info = 0
            return [p,pstar,info,it,x]
        x = np.append(x,(p1 - root))
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it,x]

def secant(x0,x1,f,tol, Nmax,y):

    count = 0
    root = 1.134724138015194
    y = np.append(y,(x1 - root))
    if (abs(f(x1) - f(x0)) == 0):
        ier = 1
        pstar = x1
        return [pstar, ier,y]
    
    while (count < Nmax):

        x2 = x1 - f(x1) * (x1 - x0)/(f(x1) - f(x0))
        y = np.append(y,(x2 - root))
        if (abs(x2 - x1) < tol):
            pstar = x2
            ier = 0
            print(count)
            return [pstar, ier,y]
        x0 = x1
        x1 = x2
        if (abs(f(x1) - f(x0)) == 0):
            pstar = x2
            ier = 1
            print(count)
            return [pstar, ier,y]
        count = count + 1
        
    pstar = x2
    ier = 1

    return [pstar, ier,y]

driver()

