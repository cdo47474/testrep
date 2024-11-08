import numpy as np
from scipy import special
import matplotlib.pyplot as plt

x = lambda x: np.pow(x,6) - x - 1
y = lambda x: 6*np.pow(x,5) - 1

# Need to plot |x_n+1 - alpha| vs |x_n - alpha| on log-log axes
alpha = 1.1347241384015194


x = np.zeros((3,1))
x[0] = 1
x[1] = 2
x[2] = 3

count = 0
# while(count < 2):
#     y1 = x[count + 1]
#     x1 = x[count]
#     plt.scatter(x1,y1)
#     count = count + 1

x = np.pow(np.e,3)
print(x)
temp = np.log(x)
print(temp)


