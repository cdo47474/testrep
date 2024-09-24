import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = x - 4*np.sin(2*x) - 3
z = 0*x

plt.plot(x, y)
plt.plot(x, z)
plt.show()