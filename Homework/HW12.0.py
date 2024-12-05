import numpy as np
import math
import time
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import special
import matplotlib.pyplot as plt
# %matplotlib inline

'''Question 2 work:'''

# A = np.array([[ 12, 10, 4], 
#      [10, 8, -5], 
#      [4, -5, 3]])


# a = np.array([[10],
#      [4]])
# e1 = np.array([[1],
#      [0]])
# w1 = a + 1*np.pow((10**2 + 4**2),0.5)*e1
    
# w = w1/norm(w1)
# wt = np.transpose(w)

# Hw = np.identity(2) - 2*np.dot(w,wt)
# # print(w1)
# # print(w)
# # print(Hw)
# H = np.array([[1, 0, 0],
#               [0, Hw[0,0], Hw[0,1]],
#               [0, Hw[1,0], Hw[1,1]]])
# A1 = np.dot(H, A)
# At = np.dot(A1,H)
# # print(At)
    
'''Question 3 work:'''
n=16
tol = 1e-7
A = np.zeros((n,n))

for i in range(1,n+1):
    
    for j in range(1,n+1):
        A[i-1,j-1] = 1/(i+j-1)


'''Power Method'''
# Guess vector: [1,1,1,1]
# v0 = np.ones((n,1))
# v1 = np.zeros((n,1))
# vold = np.zeros((n,1))
# print(v0)

# count = 0
# while(abs(vold[n-1] - v0[n-1]) > tol):
    
#     vold = v0
#     v1 = np.dot(A,v0)
#     for l in range(n):
#         temp = v1[0]
#         if(v0[l] > temp):
#             temp = v0[l]

#     v0 = v1/temp

#     count = count + 1

# print("Dominant Eigenvector: ", v0)
# print("error: ", abs(vold[n-1] - v0[n-1]))
# print("Iterations: ", count)

# a = np.dot(A,v0)
# # print(a)
# at = np.transpose(a)
# b = np.dot(at,v0)

# v0t = np.transpose(v0)
# c = np.dot(v0t,v0)
# l1 = b/c
# print("Corresponding Eigenvector: ", l1)

'''Inverse Power Method'''

Ainv = inv(A)

mu = 0.5

v0 = np.ones((n,1))
v1 = np.zeros((n,1))
vold = np.zeros((n,1))
print(v0)

count = 0
while(abs(abs(vold[n-1]) - abs(v0[n-1])) > tol):
    
    vold = v0
    aa = A - mu * np.identity(n)
    bb = inv(aa)
    v0 = np.dot(bb,v0)/ (norm(bb*v0))
    print(v0)
    

    count = count + 1
    # if (count == 25):
    #     break



print("Smallest Eigenvector: ", v0)
print("error: ", abs(abs(vold[n-1]) - abs(v0[n-1])))
print("Iterations: ", count)

a = np.dot(Ainv,v0)
# print(a)
at = np.transpose(a)
b = np.dot(at,v0)

v0t = np.transpose(v0)
c = np.dot(v0t,v0)
l1 = b/c
print("Corresponding Eigenvalue: ", l1)