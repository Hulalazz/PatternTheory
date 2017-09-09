# we aim to reproduce the graph
# where to find that 8000 sample data?

'''
Traceback (most recent call last):
  File "C:/Users/Tong Hui Kang/Dropbox/UROP for LCC/companion_ch2/page82_kurtosis_recreation.py", line 22, in <module>
    y, sr = librosa.load(librosa.util.example_audio_file())
  File "C:\Python27\lib\site-packages\librosa\core\audio.py", line 107, in load
    with audioread.audio_open(os.path.realpath(path)) as input_file:
  File "C:\Python27\lib\site-packages\audioread\__init__.py", line 96, in audio_open
    raise NoBackendError()
audioread.NoBackendError
'''


import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import time
import os
import librosa

start_time = time.time()

# read file
fileDir = os.path.dirname(os.path.realpath('__file__'))
file_name = os.path.join(fileDir, '../sounds/never ever.wav')
file_name = os.path.abspath(os.path.realpath(file_name))
ref, sample_rate = sf.read(file_name)

t_start = 02.5
t_end = 12.5
signal = ref[int(t_start * 44100): int(t_end * 44100), 0]
signal_length = len(signal)

y, sr = librosa.load(librosa.util.example_audio_file())  ###############
D = librosa.stft(y)
D_left = librosa.stft(y, center=False)
D_short = librosa.stft(y, hop_length=64)

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

# read and display code at 1220Hz


# take the derivative of the code at 1220Hz

# plot the graph

# lolz

