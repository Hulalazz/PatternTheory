# the computationally faster way

import numpy as np
import matplotlib.pyplot as plt
import samp_helper as hp
import samp_graphing as gp
import samp_toeplitz_cholesky as teo
import samp_inverse_circulant as ic
import samp_randvect as rv
import soundfile as sf
from scipy.linalg import circulant


class sampling_time(gp.graphing_multiple):
    def __init__(self, N, p, a, b):
        super(sampling_time, self).__init__(N)
        p = int(p)

        #self.ax01.set_ylim(0, int(8 * np.sqrt(self.N)))
        #self.ax02.set_ylim(-12, 12)

        #self.ax01.set_ylim(0, 5)
        #self.ax02.set_ylim(-1, 1)

        # Row/Column of inverse covariance matrix
        icovx = [0] * N
        icovx[0] = b + 2 * a
        icovx[p] = -a
        icovx[-p] = -a


        # Row/Column of covariance matrix
        covx = ic.inv_circulant(icovx)
        self.covx = np.divide(covx, covx[0])
        self.L = teo.toeplitz_cholesky_lower(N, self.covx)

        print("check")

    def updateData(self, i):
        if i > 0:
            # sample = np.random.multivariate_normal([0]*self.N, circulant(self.covx))
            sample = np.dot(self.L, np.random.randn(self.N))
            print("check")

            # To save screenshots and random variables
            # plt.savefig(r'screenshots\02_03.png', bbox_inches='tight')
            #np.savetxt("randvect/randtime{}".format(str(self.N)), sample, delimiter='\n')
            # sample = rv.randtime(self.N)

            sample = hp.convolve(sample, self.N)
            sample = hp.normalise(sample)
            sample = hp.window_std(sample, self.N)

            spectrum_post = np.absolute(np.fft.fft(sample))

            self.p011.set_data(self.x01, spectrum_post[1:int(self.N/2)])
            self.p021.set_data(self.x02, sample)
            hp.play_sound(sample, self.N)
        return self.p011, self.p021


if __name__ == '__main__':
    anima = sampling_time(4000, 4000/130.81, 1, 0.01)
    anima.start()
    plt.show()
