# a faster method to calculate the Cholesky decomposition

import numpy as np
from scipy.linalg import circulant
import samp_inverse_circulant as ic


def toeplitz_cholesky_lower(n, col):
    arr = [[0] * n for _ in range(n)]
    g = [[0] * n for _ in range(2)]

    g[0][:] = col  # G[0][0:N] contains the first row.
    g[1][:] = col  # G[1][1:N] contains the first column.

    if g[0][0] != 1:
        print("error predicted, diagonal needs to be 1")

    for entry in range(n):
        arr[entry][0] = g[0][entry]
    g[0][1:] = g[0][:-1]
    g[0][0] = 0.0

    for i in range(1,n):
        rho = - g[1][i]/g[0][i]
        div = np.sqrt((1.0 - rho)*(1.0 + rho))
        for j in range(i,n):
            g1j = g[0][j]
            g2j = g[1][j]
            g[0][j] = (g1j + rho*g2j) / div
            g[1][j] = (rho*g1j + g2j) / div
        for j in range(i,n):
            arr[j][i] = g[0][j]
        for j in range(n-1, i, -1):
            g[0][j] = g[0][j-1]
        g[0][i] = 0.0
    return arr

def bmatrix(a):
    text = r'$\left[\begin{array}{*{'
    text += str(len(a[0]))
    text += r'}c}'
    text += '\n'
    for x in range(len(a)):
        for y in range(len(a[x])):
            text += str(np.round(a[x][y],4))
            text += r' & '
        text = text[:-2]
        text += r'\\'
        text += '\n'
    text += r'\end{array}\right]$'

    print text

if __name__ == '__main__':
    N = 100
    a = 1
    b = 0.01
    p = 3

    # Row/Column of inverse covariance matrix
    icovx = [0] * N
    icovx[0] = b + 2 * a
    icovx[p] = -a
    icovx[-p] = -a


    # OFF THE SHELF METHOD
    icov = circulant(icovx)
    cov = np.linalg.inv(icov)
    cov = (1/cov[0][0]) * np.array(cov)
    L1 = np.linalg.cholesky(cov)

    # OPTIMISED METHOD
    # Row/Column of covariance matrix
    covx = ic.inv_circulant(icovx)
    covx = (1/covx[0]) * np.array(covx)
    L2 = toeplitz_cholesky_lower(N, covx)

    '''
    for row in range(0, N, p):
        plt.plot(cov[row], lw=0.2)
    plt.show()'''

    # COMPARE
    print(cov[0])
    print(covx)
    bmatrix(L1)
    bmatrix(L2)

    LX = np.add(L1, -1 * np.array(L2))
    bmatrix(LX)