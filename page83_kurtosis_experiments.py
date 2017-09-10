# experimental verification of the three kurtosis identities
# https://en.wikipedia.org/wiki/Kurtosis
# http://mathworld.wolfram.com/Kurtosis.html
# https://brownmath.com/stat/shape.htm

# The coin toss has the most platykurtic (sub-Gaussian) distribution.
# The Possion distribution is one of the most leptokurtic distribution.

# Let r.v. X be the sum of the a set of n identical and independently distributed r.v. Y

import numpy as np
from scipy.stats import kurtosis

# taking kurtosis of Y, using a larger sampler size for Y
# coin filpping
b = 500000  # as n is small, we would use a larger number b for the kurtosis of Y
# Y = np.random.random_integers(0, 10, b)  # assign a random int from 0 to k to each entry of Y
Y = np.random.poisson(1.5, b) # possion with lambda
print("mean of Y: {}".format(np.mean(Y)))
print("variance of Y: {}".format(np.var(Y)))
print("kurtosis of Y: {} \n".format(kurtosis(Y)))
# print("kurtosis of Y: {}".format(kurtosis(Y) * b / (b - 1)))  # bias/unbiased?

n = 2  # number of iid Y that make up X
m = 100000  # number of samples of X we are taking
z = 100  # z is the number of experiments

X = [0]*m  # initialising r.v. X
kurtosis_X = [0]*z  # z is the number of experiments

for k in range(z):  # perhaps can be parallelised
    for j in range(m):
        # coin flipping
        # Y = np.random.random_integers(0, 10, n)  # assign a random int from 0 to k to each entry of Y
        Y = np.random.poisson(1.5, n)  # possion with lambda
        X[j] = np.sum(Y)  # takes the sum of Y
    kurtosis_X[k] = kurtosis(X) * m / (m - 1)  # bias/unbiased sorcery

print("mean of X(Y): {} ({})".format(np.mean(X), np.mean(X)/n))
print("variance of X(Y): {} ({})".format(np.var(X), np.var(X)/n))
print("kurtosis of X: {} std_dev: {}".format(np.median(kurtosis_X), np.std(kurtosis_X)))
print("kurtosis of Y: {} std_dev: {}".format(np.median(kurtosis_X)*n, np.std(kurtosis_X)*n))  # for comparison with kurtosis of Y
