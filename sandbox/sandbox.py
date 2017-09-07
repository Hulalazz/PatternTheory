import numpy as np
import matplotlib.pyplot as plt
import x_helper as hp
import sandbox2 as gp
import soundfile as sf

a = "sounds/various_instruments_middle_C.wav"
b = "sounds/trumpet_C_major_scale.wav"
c = "sounds/toontrack_canon_grand_trimmed.wav"
d = "sounds/toontrack_canon_grand.wav"
e = "sounds/schimmel_soundtest_trimmed.wav"
f = "sounds/schimmel_soundtest.wav"
g = "sounds/river_intro.wav"
h = "sounds/piano_C_major_scale.wav"

class substitute_phase(gp.graphing_multiple):
    def __init__(self, N, ffreq, signal):
        super(substitute_phase, self).__init__(N)

        hp.play_sound(signal, self.N)
        self.to_analyse = signal

        self.spectrum = np.fft.fft(self.to_analyse)
        self.spectrum = [np.absolute(entry) for entry in self.spectrum]

        self.ax01.set_xlim(0, 1000)

        self.tw2 = [0]*N
        self.tw3 = [0] * N

    def updateData(self, i):
        if i == 0:
            self.p011.set_data(self.x01, self.spectrum[1:int(self.N/2)])
            self.time_window = hp.window_with_ref([1]*self.N, self.to_analyse)
            self.p021.set_data(self.x02, self.to_analyse)
            self.p022.set_data(self.x02, self.time_window)
        else:
            sound, actual_spectra = self.subsituting_phase()
            self.p013.set_data(self.x01, actual_spectra[1:int(self.N/2)])
            self.p021.set_data(self.x02, sound)
            self.p023.set_data(self.x02, self.tw2)
            self.p024.set_data(self.x02, self.tw3)
        return self.p011, self.p021

    def subsituting_phase(self):
        fcef = hp.complete_magnitude(self.spectrum[1:int(self.N/2)])
        sound = hp.ifft(fcef)
        self.tw3 = sound
        sound = self.window_with_ref(sound, self.to_analyse)
        hp.play_sound(sound, self.N)

        actual_spectra = np.fft.fft(sound)
        actual_spectra = [np.absolute(entry) for entry in actual_spectra]

        return sound, actual_spectra

    def window_with_ref(self, sound, signal):
        # signal is orignal, sound is generated. Need news terms.
        N = len(sound)

        block_size = int(N / 100)
        peaks = []
        for block in range(100):
            maxi = max(sound[block * block_size:(block + 1) * block_size])
            mini = min(sound[block * block_size:(block + 1) * block_size])
            peaks.append(maxi / 2 - mini / 2)
        peaks.append(peaks[-1])

        for x in range(100-1):
            for y in range(block_size):
                self.tw2[x * block_size + y + int(block_size/2)] = peaks[x] + (peaks[x + 1] - peaks[x]) / block_size * y
        for y in range(int(block_size/2 + 1)):
            self.tw2[N - y - 1] = self.tw2[N - block_size/2 - 2]
            self.tw2[y] = self.tw2[block_size/2]
        # to avoid dividing by zero
        self.tw2 = 0.99 * np.array(self.tw2) + [0.01] * N

        tw3 = np.divide([1]*N, self.tw2)

        sound = np.multiply(tw3, sound)
        #sound = np.multiply(self.time_window, sound)
        return sound



if __name__ == '__main__':
    sig, samplerate = sf.read(a)
    N = 44100
    sig = sig[8 * N : 9 * N, 0]
    anima = substitute_phase(N, 261, sig)
    anima.start()
    plt.show()