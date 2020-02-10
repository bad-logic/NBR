from django.shortcuts import render
from django.http import HttpResponse
import sounddevice as sd
import scipy.io.wavfile as wav
from keras.models import load_model
import librosa
import os
import numpy as np
import tensorflow as tf

###############################################################################
fs=16000
duration = 1  # seconds
feature_dim_1 = 20
feature_dim_2 = 21
channel = 1
###############################################################################

labels = ['noise','क','क्ष','ख','ग','घ','ङ','च','छ','ज','ज्ञ','झ','ञ','ट','ठ','ड',
'ढ','ण','त','त्र','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','स','ह']

label_num = len(labels)

global graph   #cannot call cnn model in loop so use global variable
graph = tf.get_default_graph()


################################################################################

def home(request):
    return render(request,'predict/home.html')


def record(request):
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float32')
    print ("Recording...")
    sd.wait()
    wav.write("predict.wav", fs, myrecording)
    print("Recording completed.")
    return render(request,'predict/home.html')

def predict(request):
    dict={}
    sample = wav2mfcc('predict.wav', max_len=21)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    with graph.as_default():
        model = load_model('model1_aug.h5')
        prediction = model.predict(sample_reshaped)
    print(prediction)
    print(labels)
    total_score = np.sum(prediction)
    for i in range(label_num):
        if (labels[i]=='क्ष'):
            labels[i]='xchya'
        if (labels[i]=='त्र'):
            labels[i]='tra'
        if (labels[i]=='ज्ञ'):
            labels[i]='gya'
        dict[labels[i]] = (prediction[0][i]/total_score)*100 #converting to %
    print(np.max(prediction))
    predicted = labels[np.argmax(prediction)]
    print(predicted)
    if(predicted == 'noise'):
            predicted="sorry!!! couldn't hear you"
    return render(request,'predict/prediction.html',{'values':dict,'max':predicted})

def wav2mfcc(file, max_len=21):#change max_len
    wave, sr = librosa.load(file, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc
###############################################################################
