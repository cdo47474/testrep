import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt

Mac6 = lambda x: x - np.pow(x,3)/6 + np.pow(x,5)/120

P33 = lambda x: (x - 7*np.pow(x,3)/60)/(1 + (x**2)/20)
P24 = lambda x: (x)/(1 + (x**2)/6 + (x**4)/36)
P42 = lambda x: (x - 7*np.pow(x,3)/60)/(1 + (x**2)/20)
N = 100
xeval = np.linspace(0,5,N+1)

MacEval = np.zeros(N+1)
P42eval = np.zeros(N+1)
for kk in range(N+1):
    MacEval[kk] = Mac6(xeval[kk])
    P42eval[kk] = P42(xeval[kk])

err = abs(MacEval - P42eval)

plt.figure()
plt.plot(xeval, MacEval, label= 'MacLauren')
plt.plot(xeval, P42eval,'ro-')
plt.show()

plt.figure()
plt.title('Error')
plt.semilogy(xeval, err)
plt.show()
