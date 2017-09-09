# now we are working on the discrete pdf of the random variable

import numpy as np

def print_stat_details(rv, name_of_rv):
    len_rv = len(rv)
    mean_rv = np.dot(np.arange(len_rv), rv)
    diff_from_mean_vect = np.arange(-mean_rv, -mean_rv + len_rv, 1)
    # print diff_from_mean_vect
    var_vect = np.power(diff_from_mean_vect, 2)
    var_rv = np.dot(var_vect, rv)
    kurt_vect = np.power(diff_from_mean_vect, 4)
    kurt_rv = -3 + np.dot(kurt_vect, rv) / var_rv ** 2
    print("mean of {}: {}".format(name_of_rv, mean_rv))
    print("variance of {}: {}".format(name_of_rv, var_rv))
    print("kurtosis of {}: {} \n".format(name_of_rv, kurt_rv))

# define discrete pdf of Y
Y = [0.25, 0.25, 0.25, 0.25]

# calculate mean, variance, kurtosis of Y
print_stat_details(Y, "Y")

# calculate discrete pdf of X, through convolution
X = np.convolve(Y, Y, mode="full")
print X

# calculate mean, variance, kurtosis of X
print_stat_details(X, "X")



