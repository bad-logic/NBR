# -*- coding: utf-8 -*-
# for recording the audio clip in windows
import sounddevice as sd
import scipy.io.wavfile as wav
import os

fs=16000
duration = 1  # seconds
if __name__ == '__main__':
    newpath = r'C:\Users\Roshan\Desktop\datasets'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    dataset = 1
    while(dataset == 1):
        string = input("enter random string or word:")
        label = input("enter the alphabet :" )
        newpath1 = r'C:\Users\Roshan\Desktop\datasets\{}'.format(label)
        if not os.path.exists(newpath1):
            os.makedirs(newpath1)
        for i in range(20):
            myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float32')
            print ("Recording \'{}\'{}:".format(label,i+1))
            sd.wait()
            os.chdir(newpath1)
            wav.write("{}_{}_{}.wav" .format(string,label,str(i+1)), fs, myrecording)
        dataset = int(input("hit 1 to record again, or any to exit:"))
