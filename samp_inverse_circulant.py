# code that calculate the inverse of a circulant function more quickly

import numpy as np

def inv_circulant(col):
    trans = np.fft.fft(col)
    trans_rcp = 1. / np.array(trans)
    inv = np.fft.ifft(trans_rcp)
    return inv.real

# include test code here