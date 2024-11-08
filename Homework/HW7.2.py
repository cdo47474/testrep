import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt

'''Im the goat, as seen below'''
# f = lambda x: x
# f2 = lambda x: x + 3

# a = -1
# b = 1
# N = 50


# X1 = np.linspace(a,b,N+1)

# # I want a vector of size N+1
# # Xj = np.cos(((2j - 1)*np.pi)/2*N)
# X2 = np.zeros((N+1,1))

# for j in range(N+1):
#     X2[j] = np.cos(((2*j - 1)*np.pi) / (2*N))



# y1 = f(X1)
# y2 = f2(X2)
# plt.figure()
# plt.plot(X1, y1, 'x')
# plt.plot(X2, y2, 'o')

# plt.show()

'''Same code from HW7.1, but change the interpolation nodes'''
def driver(): 

    f = lambda x: 1/(1+(10*x)**2)
    
    N = 11
    a = -1
    b = 1
    
    ''' Create interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    xint2 = np.zeros((N+1,1))

    for j in range(N+1):
        xint2[j] = np.cos(((2*j - 1)*np.pi) / (2*N))
    
    # print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
    yint2 = f(xint2)
    # print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
    # print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
    # print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    
    # print('coef = ', coef)

# No validate the code
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

    # -----------
    xeval2 = np.zeros((Neval+1,1))

    for j in range(Neval+1):
        xeval2[j] = np.cos(((2*j - 1)*np.pi) / (2*Neval))
    yeval2 = eval_monomial(xeval2,coef,N,Neval)

# exact function
    yex = f(xeval)
    
    err =  norm(yex-yeval) 
    print('err = ', err)
    
    
    '''LAGRANGE'''

    yeval_l= np.zeros(Neval+1)
    yeval_l2 = np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_l2[kk] = eval_lagrange(xeval2[kk],xint2,yint2,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
          

    


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--') 
    plt.plot(xeval,yeval_dd,'c.--') 
    plt.legend()

    plt.figure()
    plt.title('Chebyshev')
    plt.plot(xeval,fex)
    plt.plot(xeval2, yeval_l2, 'o')

    '''COOL GRAPHS'''

    # plt.figure() 
    # err_l = abs(yeval_l-fex)
    # err_dd = abs(yeval_dd-fex)
    # plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    # plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    # plt.legend()
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

    return

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
    # print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
        # print('yeval[i] = ', yeval[i])
        # print('a[j] = ', a[j])
        # print('i = ', i)
        # print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V     

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval


driver()    



