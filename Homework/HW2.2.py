import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0,2*np.pi, 1000)

for i in range (1,11):
    R = i
    dr = 0.05
    f = 2 + i
    p = 0
    x = R*(1 + dr*np.sin(f*theta + p))*np.cos(theta) 
    y = R*(1 + dr*np.sin(f*theta + p))*np.sin(theta) 
    plt.plot(x, y)

print("Done")
plt.show()