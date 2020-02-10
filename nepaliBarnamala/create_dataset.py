import sounddevice as sd
import scipy.io.wavfile as wav
import os

fs=16000
duration = 1  # seconds

def createDataset():
    newpath = r'/media/logic/Workplace/majorProject/majorProject_files/nepaliBarnamala/datasets'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    dataset = 1
    while(dataset == 1):
        string = input(print("enter any random string:"))
        label = input(print("enter label of dataset:"))
        newpath1 = r'/media/logic/Workplace/projects/majorProject/nepaliBarnamala/datasets/{}'.format(label)
        if not os.path.exists(newpath1):
            os.makedirs(newpath1)
        for i in range(20):
            myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float32')
            print ("Recording \'{}\'{}:".format(label,i+1))
            sd.wait()
            os.chdir('/media/logic/Workplace/projects/majorProject/nepaliBarnamala/datasets/{}'.format(label))
            wav.write("{}_{}_{}.wav" .format(string,label,str(i+1)), fs, myrecording)

        dataset = int(input(print("Enter 1 to continue or  any key to stop:")))

createDataset()
