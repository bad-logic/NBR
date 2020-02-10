import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from matplotlib import cm
from tqdm import tqdm
import os
############################################################################
DATA_PATH = "/media/logic/Workplace/projects/majorProject/nepaliBarnamala/not_aug_datasets/"
##############################################################################

def save_data_to_array(path=DATA_PATH, max_len=21):
    labels = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' +
                    wavfile for wavfile in os.listdir(path + '/' + label)]
        #print(wavfiles)
        for wavfile in tqdm(wavfiles,
                            "Saving vectors of label - '{}'".format(label)):
            mfcc_coeff = mfcc(wavfile, max_len = max_len)
            #make_image(mfcc, label)#generate image from mfcc
            mfcc_vectors.append(mfcc_coeff)
        np.save(label + '.npy', mfcc_vectors)

##############################################################################

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    return labels

##############################################################################

def mfcc(wave, max_len=21):
    wave, sr = librosa.load(wave, mono=True, sr=None)
    wave_freq = len(wave)
    frame_size = 0.05  #50ms (standard time frame is 25ms. typically  Range(20-40 ms frames))
    no_of_frames =   20
    samples = int(wave_freq/no_of_frames) # or ( wave_freq * frame_size)
    #print(samples)

    #pre emphasising the wave
    pre_wave = np.append(wave[0],wave[1:]-(0.97*wave[:-1]))

    mfc = []

    for i in range(20):
        index = []
        for ind in range(i*samples,(i+1)*samples):
            index.append(ind)
            #print(index)
        frame = pre_wave[index]
        window = frame * np.hamming(samples)
        #Fast fourier transform
        N = 512
        fft = np.absolute(np.fft.rfft(window, N))#making -ve values +ve using absolute
        #calculating power spectrum
        power_spec = ((1.0/N)*((fft)**2))
        #mel-frequency wrapping
        nfilt = 40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sr/2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((N + 1) * hz_points / sr)
        fbank = np.zeros((nfilt, int(np.floor(N / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(power_spec, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        #apply DCT
        coeff = dct(filter_banks, type=2, norm='ortho')
        mfc.append(coeff[1:13])


    mfc = np.array(mfc)
    if (max_len > mfc.shape[1]):
        pad_width = max_len - mfc.shape[1]
        mfc = np.pad(mfc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfc = mfc[:, :max_len]
    return mfc
    #print(mfc)
    #fig, ax = plt.subplots()
    #fig = ax.imshow(mfc, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    #plt.axis('off')
    #plt.show()
##############################################################################
if __name__ == "__main__":

    #save_data_to_array()
    #mfcc = mfcc("ka.wav", max_len=21)
    #print(mfcc)
    #fig, ax = plt.subplots()
    #fig = ax.imshow(mfcc, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    #plt.axis('off')
    #plt.show()
    print("nothing assigned")
