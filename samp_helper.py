# functions to assist in code that generate multivar samples

import numpy as np
import sounddevice as sd
import samp_randvect as rv
import matplotlib.pyplot as plt
import time


def complete_magnitude(magnitude):
    # the input is the absolute value, excludes zero at 0 - total of N/2 - 1
    # random = np.random.random(22049)
    random = rv.random(22049)  # for using the same random vector for the screenshots
    phase = (2 * np.pi) * np.array(random)
    reals = np.multiply(magnitude, np.sin(np.array(phase)))
    imags = np.multiply(magnitude, np.cos(np.array(phase)))

    fcef = complete_realimags(reals, imags)
    return fcef


def complete_realimags(reals, imags):
    # the input excludes zero at 0 - total of N/2 - 1 entries
    reals = np.concatenate([reals, reals[::-1]])
    reals = np.insert(reals, int(len(reals) / 2), 0)
    reals = np.insert(reals, 0, 0)

    imags = np.concatenate([imags, -1 * np.array(imags[::-1])])
    imags = np.insert(imags, int(len(imags) / 2), 0)
    imags = np.insert(imags, 0, 0)

    fcef = np.vectorize(complex)(reals, imags)
    return fcef


def ifft(fcef):
    signal = np.fft.ifft(fcef)
    signal_real = [entry.real for entry in signal]
    return signal_real


def convolve(signal, N):
    modulation = [(x / (N/400.)) * np.exp(-x / (N/40.)) for x in range(1, int(N/2))]
    filter = complete_realimags(modulation, [0] * int(N/2 - 1))
    filter_x = np.real(np.fft.ifft(filter))
    #signal = np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(filter_x)))
    signal = np.convolve(filter_x, signal, mode="same")
    # not the same :(
    print len(signal)
    ''' this is used to plot for the convolve factor
                N = self.N
            modulation = [(x / (N / 400.)) * np.exp(-x / (N / 40.)) for x in range(1, int(N / 2))]
            filter = hp.complete_realimags(modulation, [0] * int(N / 2 - 1))
            filter_x = np.real(np.fft.ifft(filter))
            sample = filter_x
    '''
    return signal


def normalise(signal):
    max_value = max(max(signal), -min(signal))
    signal[:] = [x / max_value for x in signal]
    return signal


def window_std(audio, N):
    N = len(audio)
    window = [0]*N
    for t in range(30, N):
        window[t] = np.exp(-t / (N/4.)) * np.arctan(t / (N/400.))
    audio = normalise(audio)
    audio = np.multiply(audio, window)
    return audio


def envelope_fitting(audio, ref_time_envelope):
    audio = normalise_through(audio)
    audio = np.multiply(audio, ref_time_envelope)
    return audio


def normalise_through(audio):
    sample_time_profile = calculate_time_envelope(audio, True)
    audio = np.divide(audio, sample_time_profile)
    return audio


def calculate_time_envelope(audio, zero_pad=False):
    N = len(audio)
    num_blocks = 100
    # number of blocks is arbitrary chosen. Ideally, the ffreq should be larger than the number of block.
    # yet to test round-off errors
    block_size = int(N / num_blocks)

    peaks = []
    for block in range(num_blocks):  # measuring the peaks
        maxi = max(audio[block * block_size:(block + 1) * block_size])
        mini = min(audio[block * block_size:(block + 1) * block_size])
        peaks.append(maxi / 2 - mini / 2)

    window = [0] * N
    for x in range(num_blocks - 1):  # connecting the peaks together
        for y in range(block_size):
            window[x * block_size + y + int(block_size / 2)] = peaks[x] + (peaks[x + 1] - peaks[x]) / block_size * y
    for y in range(int(block_size / 2 + 1)):
        window[N - y - 1] = window[N - block_size / 2 - 2]
        window[y] = window[block_size / 2]
    # to avoid dividing by zero in normalise_throughout
    if zero_pad:
        window = 0.99 * np.array(window) + [0.01] * N
    return window


def play_sound(audio, sampling_rate):
    sd.play(audio, sampling_rate)


