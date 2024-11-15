import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt
# %matplotlib inline

x,y = np.meshgrid(np.linspace(0,0.3,10),np.linspace(0,0.3,10))

u = x
v = -y

plt.quiver(x,y,u,v)
plt.show()
    

    
    


