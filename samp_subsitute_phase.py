# basically take the spectrum and replace with random phases
# then make it sound similar to the original by modifying the window

import numpy as np
import matplotlib.pyplot as plt
import samp_helper as hp
import samp_graphing_multiple as gp
import soundfile as sf


class substitute_phase(gp.graphing_multiple):
    def __init__(self, N, ffreq, reference):
        super(substitute_phase, self).__init__(N)

        hp.play_sound(reference, self.N)
        self.reference = reference

        self.spectrum_ref = np.absolute(np.fft.fft(self.reference))

        self.ax01.set_xlim(0, 4000)
        self.time_envelope = hp.calculate_time_envelope(self.reference)

    def updateData(self, i):
        if i == 0:
            self.p011.set_data(self.x01, self.spectrum_ref[1:int(self.N/2)])
            self.p021.set_data(self.x02, self.reference)
            self.p022.set_data(self.x02, self.time_envelope)
        else:
            plt.savefig(r'screenshots\01_04.png', bbox_inches='tight')
            sample, spectra_post = self.subsituting_phase()
            self.p013.set_data(self.x01, spectra_post[1:int(self.N/2)])
            self.p021.set_data(self.x02, sample)
        return self.p011, self.p021

    def subsituting_phase(self):
        fcef = hp.complete_magnitude(self.spectrum_ref[1:int(self.N/2)])
        sample = hp.ifft(fcef)
        #sample = hp.envelope_fitting(sample, [1] * 44100)
        sample = hp.envelope_fitting(sample, self.time_envelope)
        hp.play_sound(sample, self.N)

        # spectra_post means the spectrum after processing in time domain
        spectra_post = np.absolute(np.fft.fft(sample))

        return sample, spectra_post

if __name__ == '__main__':
    sig, samplerate = sf.read("sounds/recorded_piano_middle_C.wav")
    #sig, samplerate = sf.read("sounds/various_instruments_middle_C.wav")
    N = 44100
    sig = sig[0 * N: 1 * N, 0]
    anima = substitute_phase(N, 261, sig)
    anima.start()
    plt.show()