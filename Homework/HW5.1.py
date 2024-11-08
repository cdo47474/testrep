import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt

x0 = np.array([1,1,1])


Nmax = 100
tol = 1e-10

for its in range(Nmax):

    F = x0[0]**2 + 4*x0[1]**2 + 4*x0[2]**2 - 16
    fx = 2*x0[0]
    fy = 8*x0[1]
    fz = 8*x0[2]
    gradf = np.array([fx,fy,fz])
    frac = F/(fx**2 + fy**2 + fz**2)

    x1 = x0 - (gradf).dot(frac)
    
    
    if(abs(F) < tol):
        
        break
    
    # error:
    print(norm(x1-x0))

    x0 = x1

print("Answer: ", x1)
    
    