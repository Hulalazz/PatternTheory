# the direct way

import numpy as np
import matplotlib.pyplot as plt
import samp_helper as hp
import samp_graphing as gp
import samp_toeplitz_cholesky as teo
import samp_inverse_circulant as ic
from scipy.linalg import circulant


class sampling_time(gp.graphing_multiple):
    def __init__(self, N, p, a, b):
        super(sampling_time, self).__init__(N)
        self.ax01.set_ylim(0, int(50 * np.sqrt(self.N)))
        self.ax02.set_ylim(-25, 25)

        p = int(p)

        # Row/Column of inverse covariance matrix
        icovx = [0] * N
        icovx[0] = b + 2 * a
        icovx[p] = -a
        icovx[-p] = -a

        # Row/Column of covariance matrix
        self.covx = ic.inv_circulant(icovx)
        #self.covx = np.divide(covx, self.covx[0])
        #self.L = teo.toeplitz_cholesky_lower(N, self.covx)

    def updateData(self, i):
        if i > 0:
            plt.savefig(r'screenshots\02a_1000_40_1_01.png', bbox_inches='tight')
            #sample = np.dot(self.L, np.random.rand(self.N))
            sample = np.random.multivariate_normal([0]*self.N, circulant(self.covx))

            #sample = hp.convolve(sample, self.N)
            #sample = hp.normalise(sample)
            #sample = hp.window_std(sample, self.N)

            spectrum_post = np.absolute(np.fft.fft(sample))

            self.p011.set_data(self.x01, spectrum_post[1:int(self.N/2)])
            self.p021.set_data(self.x02, sample)
        return self.p011, self.p021


if __name__ == '__main__':
    anima = sampling_time(1000, 40, 1, 0.1)
    anima.start()
    plt.show()
