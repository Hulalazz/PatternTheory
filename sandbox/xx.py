import numpy as np
import matplotlib.pyplot as plt
import x_helper as hp

N = 44100

modulation = [(x / 10.) * np.exp(-x / 200.) for x in range(1, int(N / 2))]
filter = hp.complete_realimags(modulation, [0] * int(N / 2 - 1))
filter_x = np.fft.ifft(filter)

plt.plot(np.real(filter_x))
plt.show()

np.set_printoptions(threshold=np.nan)

print(np.random.randn(44100))

