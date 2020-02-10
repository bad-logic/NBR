import librosa
import numpy as np
#from matplotlib import pyplot as plt
from scipy.fftpack import dct

##############################################################################

def pre_emphasis(wave):
    pre_wave = np.append(wave[0],wave[1:]-(a*wave[:-1]))
    return pre_wave

##############################################################################

def frame(wave,start,stop):
    index = []
    #return wave[start:stop] #numpy arrays can't be sliced this way
    for i in range(start,stop):
        index.append(i)
    return wave[index]   # wave[1,2,3,4,5,6,7] return corresponding values of this indices 1,2,3,4,5,6,7

###############################################################################

def window(frame):
    window = frame * np.hamming(samples)
    return window
##############################################################################

def fft(window):
    fft = np.absolute(np.fft.rfft(window, N))#making -ve values +ve using absolute
    return fft                               # becoz absolute in power spec

###############################################################################

def power_spectrum(fft):
    power_spec = ((1.0/N)*((fft)**2))
    return power_spec

###############################################################################

def Mel_freq(power_spec):
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
        filter_banks = 20 * np.log10(filter_banks)
        return filter_banks

#################################################################################

def discrete_cosine_transform(filter_banks):
    mfcc = dct(filter_banks, type=2, norm='ortho')
    #print(mfcc[1:13])

    return mfcc[1:13]


###############################################################################

if __name__ == "__main__":

    N = 512
    nfilt = 40
    a = 0.97

    mfcc = []
    print(mfcc)


    #librosa loads wave as a numpy array
    wave, sr = librosa.load("ka.wav", mono=True, sr=None)#step1

    wave_freq = len(wave)
    frame_size = 0.05  #50ms (standard time frame is 25ms. typically  Range(20-40 ms frames))
    no_of_frames =   20
    samples =int(wave_freq/no_of_frames) # or ( wave_freq * frame_size)

    print('wave_freq:{}'.format(wave_freq))
    print('frame_size:{} seconds'.format(frame_size))
    print("samples:{}".format(samples)) #samples per frame


    pre_wave = pre_emphasis(wave) # step2
    print('emphasised wave')
    print(pre_wave)
    for i in range(no_of_frames):
        print("frame no:{}".format(i+1))
        start = i * samples
        print("start:{}".format(start))
        stop = (i+1) * samples
        print("stop:{}".format(stop-1))
        #####################################################################
        ############       frame         ########################
        #index = []
        #return wave[start:stop] #numpy arrays can't be sliced this way
        #for ind in range(start,stop):
        #    index.append(ind)
        #frame = pre_wave[index]   # wave[1,2,3,4,5,6,7] return corresponding values of this indices 1,2,3,4,5,6,7

        frame = frame(pre_wave,start,stop)# step3
        print('frame{}'.format(i+1))
        print(frame)
        #####################################################################
        window = window(frame)             # step4
        print('window{}'.format(i+1))
        print(window)
        fft = fft(window)                  #step5
        print('fft{}'.format(i+1))
        print(fft)
        spectrum = power_spectrum(fft)     #step6
        print('spectrum{}'.format(i+1))
        print(spectrum)
        filter_coeff = Mel_freq(spectrum) #step7
        print('filter_coeff{}'.format(i+1))
        print(filter_coeff)
        mel_coeff = discrete_cosine_transform(filter_coeff) #step8
        print('mel_coeff{}'.format(i+1))
        print(mel_coeff)
        mfcc.append(mel_coeff)
        print(mfcc)

    print(mfcc)
