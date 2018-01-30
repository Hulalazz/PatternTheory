import numpy as np
import matplotlib.pyplot as plt
import samp_helper as hp
import samp_graphing_multiple as gp
import soundfile as sf
import samp_randvect as rv


class substitute_phase_gauss(gp.graphing_multiple):
    def __init__(self, N, ffreq, reference, ihmrc):
        super(substitute_phase_gauss, self).__init__(N)

        hp.play_sound(reference, N)
        self.reference = reference
        self.time_profile = hp.calculate_time_envelope(self.reference)

        self.spectrum_ref = np.fft.fft(reference)
        self.spectrum_ref = [np.absolute(entry) for entry in self.spectrum_ref]

        self.ax01.set_xlim(0, 5000)

        self.freq_profile = self.calc_freq_profile(self.spectrum_ref, ffreq, ihmrc)
        a = 1; b = 0.0001; s = 0.01
        # These are arbitrarily chosen, perhaps could be learnt
        self.sine = [s * (b + 4*a*((np.sin(np.pi * freq / ffreq)) ** 2)) ** -0.5 for freq in range(int(self.N/2) - 1)]
        self.spectra = np.multiply(self.freq_profile, self.sine)

        self.spectra_gauss = []
        self.spectra_post = []
        self.sample = [] * N

    def calc_freq_profile(self, spectrum_ref, ffreq, ihmrc):
        # requires knowledge of the fundamental frequency, rather specific to middle C
        # inharmonicity is ignored
        peaks = [0]
        for x in range(1, int(self.N/(ffreq*2) - 1)):
            peaks.append(max(spectrum_ref[int(x*ffreq)-100:int(x*ffreq)+100]))

        freq_profile = [0] * int(self.N/2)
        for x in range(int(self.N / (ffreq * 2) - 2)):
            for y in range(int(ffreq) + 1):
                freq_profile[int(x * ffreq) + y] = peaks[x] + (y * (peaks[x + 1] - peaks[x]) / ffreq)

        return freq_profile[:int(self.N/2)-1]

    def updateData(self, i):
        if i == 0:
            self.p011.set_data(self.x01, self.spectrum_ref[1:int(self.N/2)])
            self.p012.set_data(self.x01, self.freq_profile)
            #self.p013.set_data(self.x01, self.spectra) # need to have better names
            self.p022.set_data(self.x02, self.time_profile)
            self.p021.set_data(self.x02, self.reference)
        else:
            plt.savefig(r'screenshots\04_03.png', bbox_inches='tight')
            self.subsituting_phase()
            self.p013.set_data(self.x01, self.spectra_gauss)
            #self.p014.set_data(self.x01, self.spectra_post[1:int(self.N/2)])
            self.p021.set_data(self.x02, self.sample)
        return self.p011, self.p012, self.p013, self.p021, self.p022

    def subsituting_phase(self):
        #random = np.random.randn(int(self.N/2-1))
        random = rv.randn(22049)

        self.spectra_gauss = np.multiply(self.spectra, random)

        fcef = hp.complete_magnitude(self.spectra_gauss)
        self.sample = hp.ifft(fcef)

        self.sample = hp.envelope_fitting(self.sample, self.time_profile)
        hp.play_sound(self.sample, self.N)

        self.spectra_post = np.absolute(np.fft.fft(self.sample))

if __name__ == '__main__':
    ref, sample_rate = sf.read('sounds/recorded_piano_middle_C.wav')
    N = sample_rate
    reference = ref[0 * N: 1 * N, 0]
    anima = substitute_phase_gauss(N, 261.62/2, reference, 1)
    anima.start()
    plt.show()