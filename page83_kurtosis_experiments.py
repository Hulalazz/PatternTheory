# experimental verification of the three kurtosis identities
# https://en.wikipedia.org/wiki/Kurtosis
# http://mathworld.wolfram.com/Kurtosis.html
# https://brownmath.com/stat/shape.htm

# The coin toss has the most platykurtic (sub-Gaussian) distribution.
# The Possion distribution is one of the most leptokurtic distribution.

# Let r.v. X be the sum of the a set of n identical and independently distributed r.v. Y


import numpy as np
from scipy.stats import kurtosis

n = 10  # number of iid Y that make up X
m = 10000  # number of samples of X we are taking

X = [0]*m  # initialising r.v. X

for j in range(m):
    Y = np.random.random_integers(0, 1, n)  # assign 0 or 1 to each entry of Y
    X[j] = np.sum(Y)  # takes the sum of Y

b = 100000  # as n is small, we would use a larger number b for the kurtosis of Y
Y = np.random.random_integers(0, 1, b)

kurtosis_X = kurtosis(X) * m/(m-1)  # bias/unbiased sorcery
kurtosis_Y = kurtosis(Y) * b/(b-1)

print kurtosis_X  # inaccurate perhaps due to closeness
print kurtosis_X*n
print kurtosis_Y

