import numpy as np
from scipy import special
import matplotlib.pyplot as plt


h = 0.01 * 2. **(-np.arange(0, 10))

xa = np.pi/2

f = lambda x: np.cos(x)

fprime1 = (f(xa + h) - f(xa))/h

print("Approximation with first f'(x)")
print(fprime1)

fprime2 = (f(xa + h) - f(xa - h))/(2*h)

print("--------------")
print("Approximation with second f'(x)")
print(fprime2)