import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt
# %matplotlib inline

x = np.linspace(-1,3,100)
plt.figure(figsize=(10,5))
for i in range(-2,4):
    m = 2**i
    I = -(m**2)/4
    
    f = lambda x: m*x + I
    plt.plot(x, f(x))
# f1 = lambda x: (np.e)**x - 1
# plt.plot(x,f1(x))
plt.show()
