from __future__ import print_function

import sys

import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkContext
from copy import copy, deepcopy

np.random.seed(42)
p = 0.01    # Probability of an element of the matrix being observed
M = 1000    # The matrix is of dimension M*U
U = 5000
F = 10      # Maximum possible rank of the matrix
ITERATIONS = 12     # T defined in the paper
partitions = 8      # TODO make it 24
N = 2*ITERATIONS + 1

def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))


def update(i, mat, ratings, current, loc):
    vec = np.nonzero(ratings[i,:])[1]
    uu = len(vec)
    if uu == 0:
        return current.T
    XtX = mat[vec,:].T * mat[vec,:] #matrix(np.dot(mat[vec,:].T,mat[vec,:])) #
    Xty = mat[vec,:].T * ratings[i, vec].T
    if len(np.nonzero(XtX)[0]) == 0: #TODO it shouldn't occur
        return current.T
    if np.linalg.cond(XtX) < 1/sys.float_info.epsilon:  #If can't be inverted
        return np.linalg.solve(XtX, Xty)
    return current.T

sc.stop()
sc = SparkContext(appName="AltMinComplete")

print("Running AltMinComplete with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
      (M, U, F, ITERATIONS, partitions))

Rcomplete = matrix(rand(M, F)) * matrix(rand(U, F).T)
R = matrix(np.zeros(shape=(M,U)))
length = R.shape[0]
width = R.shape[1]

thr = 1 - p

for i in range(length):
    for j in range(width):
        if rand(1)[0] > thr:
            R[i,j] = Rcomplete[i,j]

Dis = np.zeros(shape=(N,M,U))
for i in range(length):
    for j in range(width):
        randomNo = rand(1)[0]
        for k in range(N):
            if randomNo <= (k+1)/(N*1.0):
                Dis[k,i,j] = R[i,j]
                break

u,s,v = np.linalg.svd(Dis[0]/p)
ms = matrix(u[:,range(F)])
us = matrix(v[range(F),:].T)

Rb = sc.broadcast(Dis)
msb = sc.broadcast(ms)
usb = sc.broadcast(us)

j = 0
for i in range(ITERATIONS):
    j = j+1
    us = sc.parallelize(range(U), partitions) \
           .map(lambda x: update(x, msb.value, matrix(Rb.value[j]).T, usb.value[x, :], j)) \
           .collect()
    us = matrix(np.array(us)[:, :, 0])
    usb = sc.broadcast(us)
    j = j+1
    ms = sc.parallelize(range(M), partitions) \
           .map(lambda x: update(x, usb.value, matrix(Rb.value[j]), msb.value[x, :], j)) \
           .collect()
    # collect() returns a list, so array ends up being
    # a 3-d array, we take the first 2 dims for the matrix
    ms = matrix(np.array(ms)[:, :, 0])
    msb = sc.broadcast(ms)
    error = rmse(Rcomplete, ms, us)
    print("Iteration %d:" % i)
    print("\nRMSE: %5.10f\n" % error)

sc.stop()
