# import mypkg.my2DPlotB
import numpy as np
import math
from numpy.linalg import inv 
import matplotlib.pyplot as plt


def driver():
    
    f = lambda x: 1/(1+(np.pow(10*x, 2)))
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 3
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    # plt = mypkg.my2DPlotB(xeval,fex)
    # plt.figure()
    # plt.plot(xeval,fex)
    # plt.plot(xeval,yeval)
     
     
    # err = abs(yeval-fex)
    # plt.figure()
    # plt.plot(xeval,err)
    # plt.show()

    
    
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
    n=20
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''

        '''temporarily store your info for creating a line in the interval of 
        interest'''
        a1 = xint[jint]
        fa = f(a1)
        
        b1 = xint[jint+1]
        fb1 = f(b1)
        
        

        # Need to make a line between (a1,fa1) and (b1, fb1)
        # Slope:
        m = (fb1-fa)/(b1-a1)
        f1 = lambda x: m*(x - a1) + fa
        

        # Interval a1 to b1
        xind = np.linspace(a1,b1,n)
        yeval = f1(xind) 
        plt.plot(xind, yeval)
    
    plt.show()
    return yeval
    
  
    
    
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
