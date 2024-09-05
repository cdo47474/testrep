import numpy as np
import matplotlib.pyplot as plt
x = [1,2,3]
x = 3*x
print(x)
y = np.array([1,2,3])
print(y)
print('This is 3y', 3*y)
X = np.linspace(0, 2 * np.pi, 100) 
Ya = np.sin(X)
Yb = np.cos(X)

x1 = np.linspace(0, 10, 100)
y1 = np.arange(0, 10, 0.1)

plt.plot(X, x1)
plt.plot(X, y1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

