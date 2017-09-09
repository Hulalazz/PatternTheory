# sample amplitude of gaussian, give it random phases
# arbitrary function to change the size of the harmonics
# then ifft, and cast a time window (from arbitrary function)

import numpy as np
import matplotlib.pyplot as plt
import samp_helper as hp
import samp_graphing_multiple as gp
import samp_randvect as rv
import soundfile as sf


class gaussian_freq(gp.graphing_multiple):
    def __init__(self, N, ffreq, a, b):
        super(gaussian_freq, self).__init__(N)
        self.a = a
        self.b = b
        self.ffreq = ffreq
        self.freq_profile = [(x / 10.) * np.exp(-x / 400.) for x in range(1, int(self.N/2))]
        self.ax01.set_xlim(0, 6000)
        self.ax01.set_ylim(0, 2000)

        ref, samplerate = sf.read("sounds/recorded_piano_middle_C.wav")
        ref = ref[0 * N: 1 * N, 0]
        self.time_envelope = hp.calculate_time_envelope(ref)

    def updateData(self, i):
        if i > 0:
            plt.savefig(r'screenshots\03_01.png', bbox_inches='tight')
            sample, spectrum = self.gaussian_note_by_freq(self.N, self.ffreq, self.a, self.b)
            self.p011.set_data(self.x01, spectrum)
            self.p012.set_data(self.x01, (1/self.b) * np.array(self.freq_profile))
            self.p021.set_data(self.x02, sample)
        return self.p011, self.p012, self.p021

    def gaussian_note_by_freq(self, N, ffreq, a, b):

        # variance
        spectrum = [(b + 4*a*((np.sin(np.pi*freq/ffreq))**2)) ** -0.5 for freq in range(1, int(N/2))]
        spectrum = np.multiply(spectrum, rv.randn(22049))
        #spectrum = np.sqrt(self.N) * np.array(spectrum)
        spectrum = self.frequency_modulation(spectrum)

        fcef = hp.complete_magnitude(spectrum)
        sample = hp.ifft(fcef)
        #sample = (1/np.sqrt(self.N)) * np.array(sample)
        sample = hp.normalise(sample)
        sample = hp.envelope_fitting(sample, self.time_envelope)
        hp.play_sound(sample, N)

        spectrum_post = np.absolute(np.fft.fft(sample))
        #spectrum_post = np.sqrt(self.N) * np.array(spectrum_post)

        return sample, spectrum_post[1:int(self.N/2)]

    def frequency_modulation(self, spectrum):
        #modulation = [np.arctan(x/100.) * np.exp(-x/200.) for x in range(1,n)]
        modulation = self.freq_profile
        spectrum = np.multiply(np.abs(spectrum), modulation)
        return spectrum


if __name__ == '__main__':
    anima = gaussian_freq(44100, 261.65, 1, 0.01)
    anima.start()
    plt.show()