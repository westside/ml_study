import scipy.io.wavfile as wav

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import glob, os
import stempeg

from os import listdir
from os.path import isfile, join
from matplotlib import interactive

train_path = "./dataset/DSD100subset/Mixtures/Dev/"
original_path = "./dataset/DSD100subset/Sources/Dev/"

print(listdir(train_path))


def stfp_from_file(path, label) :
    fs, x = wav.read(path)
    x = x[:, 0]
    print (x.shape)

    window_time = 0.05
    overlap_time = 0.03

    num_sample = int(fs * window_time)
    num_overlap = int(fs * overlap_time)
    f, t, Zxx = signal.stft(x, fs, nperseg=num_sample, noverlap=num_overlap)
    amp = 0
    max_value = 0
    for val in np.abs(Zxx):
        max_value = max(max_value, max(val))

    amp = 300
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude : ' + label)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    # plt.show()

    return f, t, Zxx, max_value


for directory in listdir(train_path):
    print(original_path + directory)
    base_path = original_path + directory + '/bass.wav'
    drum_path = original_path + directory + '/drums.wav'
    other_path = original_path + directory + '/other.wav'
    vocal_path = original_path + directory + '/vocals.wav'
    mixture_path = train_path + directory + '/mixture.wav'

    fMixture, tMixture, ZxxMixture, max_value = stfp_from_file(mixture_path, 'mixture')
    fBass, tBass, ZxxBass, _ = stfp_from_file(base_path, 'bass')
    fDrum, tDrum, ZxxDrum, _ = stfp_from_file(drum_path, 'drum')
    fOther, tOther, ZxxOther, _ = stfp_from_file(other_path, 'other')
    fVocal, tVocal, ZxxVocal, _ = stfp_from_file(vocal_path, 'vocal')