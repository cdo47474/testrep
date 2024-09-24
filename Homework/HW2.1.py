import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0,2*np.pi, 1000)
R = 1.2
dr = 0.1
f = 15
p = 0

x = R*(1 + dr*np.sin(f*theta + p))*np.cos(theta) 
y = R*(1 + dr*np.sin(f*theta + p))*np.sin(theta) 

plt.plot(x, y)
plt.show()