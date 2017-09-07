# Re-creating of the kurtosis graphs in page 82
# It is different though, we didn't do STFT to calculate the data that is to be plotted.

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import time
import os
from scipy.stats import kurtosis

# start_time = time.time()

# read file
fileDir = os.path.dirname(os.path.realpath('__file__'))
file_name = os.path.join(fileDir, '../sounds/never ever.wav')
file_name = os.path.abspath(os.path.realpath(file_name))
ref, sample_rate = sf.read(file_name)

t_start = 02.5
t_end = 115.5
signal = ref[int(t_start * 44100): int(t_end * 44100), 0]
signal_length = len(signal)

# add noise so that silent parts will not give ambiguous values
# signal = np.add(signal, 0.001*np.random.randn(len(signal)))
# sd.play(signal, sample_rate)

# taking absolute (L1 norm or L2 norm)
signal_square = np.multiply(signal, signal)  # L2 norm
# signal_square = signal
# signal_square = np.absolute(signal)  # L1 norm

# the size/width of the window that is used to sum
# the book uses a certain frequency which is the result from the STFT.
window_size = 100
number_of_windows = len(signal) / window_size

# calculating the "energy" of each window
# not according to the book though
window_type = 'rect'
# rect, trig, or sin2

if window_type == 'rect':
    energy = [np.sum(signal_square[x:x + window_size]) for x in range(0, signal_length - window_size, window_size)] # rectangular window
elif window_type == 'sin2':
    window_function = [(np.sin(np.pi * x / window_size)) ** 2 for x in range(window_size)]
    energy = [np.sum(np.multiply(signal_square[x:x + window_size], window_function)) for x in range(signal_length - window_size)]
    #energy = 1 / (float(window_size)) ** (3.0 / 4.0) * np.array(energy)  # maybe not necessary
elif window_type == 'trig':
    window_function = [1.0 - np.absolute(2*x / window_size - 1.0) for x in range(window_size)]
    energy = [np.sum(np.multiply(signal_square[x:x + window_size], window_function)) for x in range(signal_length - window_size)]
    #energy = 1 / (float(window_size)) ** (3.0 / 4.0) * np.array(energy)  # maybe not necessary

# scaling down arbitrarily, shouldn't matter
energy = 1.0 / (float(window_size)) ** (1.0 / 4.0) * np.array(energy)

# energy_noise = 0  # adding noise to energy sequence, to "smoothen"
# if energy_noise != 0:
#     energy = np.add(energy, energy_noise*np.random.randn(len(energy)))

# plt.plot(energy)
# plt.show()

derivative = [(energy[x+1] - energy[x]) for x in range(len(energy) - 1)]
# difference between the energy of each of the snippet
# this is what we are interested in

# plt.plot(derivative)
# plt.show()

# derivative = energy  # for experimental purposes

'''
Plotting the distribution curve
'''

# Calculation of mean and variance
mean_mean = np.mean(derivative)  # mean
std_std = np.std(derivative)  # std_dev
print "mean: {}".format(mean_mean)
print "stddev: {}".format(std_std)

# Standardising mean and variance
derivative = np.add(derivative, [-mean_mean]*len(derivative))
derivative = 1.0/std_std * np.array(derivative)
# print "mean: {}".format(mean_mean)
# print "stddev: {}".format(std_std)

# building a histogram
number_of_bars = 999
histogram = [0.0]*(number_of_bars)
extreme = max(np.absolute(derivative))
interval = (2.*extreme)/number_of_bars
histogram_entry = 0

# populating the histogram
for x in np.arange(-extreme + interval/2.0, extreme + interval/2.0, interval):
    for entry in derivative:
        if (entry > x) and (entry < interval + x):
            histogram[histogram_entry] += 1.0/(len(derivative)*interval)
            # possible float/integer error
    histogram_entry += 1

print "area under graph: {}".format(np.sum(histogram)*interval)
# print "area under graph: {}".format(interval*np.sum([1/(std_std*np.sqrt(2*np.pi))*
#           np.exp(-(x - mean_mean)**2 / (2*(std_std)**2))
#           for x in np.arange(-extreme, extreme, interval)]))

# plotting histogram against normal distribution
plt.plot(np.arange(-extreme + interval/2.0, extreme + interval/2.0, interval),
         histogram, lw=0.8)
plt.plot(np.arange(-extreme + interval/2.0, extreme + interval/2.0, interval),
         histogram[::-1], lw=0.1)
plt.plot(np.arange(-extreme, extreme, interval),
         [1/(std_std*np.sqrt(2*np.pi))*
          np.exp(-(x - mean_mean)**2 / (2*(std_std)**2))
          for x in np.arange(-extreme, extreme, interval)])

# taking the log of the histogram
histogram = [np.log(entry + 0.1/len(derivative)) for entry in histogram]

# plotting histogram against log-normal distribution
plt.plot(np.arange(-extreme + interval/2.0, extreme + interval/2.0, interval),
         histogram, lw=0.8)
plt.plot(np.arange(-extreme + interval/2.0, extreme + interval/2.0, interval),
         histogram[::-1], lw=0.1)
plt.plot(np.arange(-extreme, extreme, interval),
         [-(x - mean_mean)**2 / (2*(std_std)**2)
          - np.log(std_std*np.sqrt(2*np.pi))
          for x in np.arange(-extreme, extreme, interval)])

derivative = np.random.randn(10000)
mean_mean = np.mean(derivative)  # mean
std_std = np.std(derivative)  # std_dev
meenus = np.add(derivative, [-mean_mean]*len(derivative))
meenus_4 = np.sum(np.power(meenus, 4))
print "kurtosis: {}".format(-3 + meenus_4/(len(derivative)*std_std**4))  # excess kurtosis
print "kurtosis: {}".format(kurtosis(derivative))  # Fisher's  definition (normal ==> 3.0)

ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_xlim([-2.5*std_std, 2.5*std_std])
plt.show()
