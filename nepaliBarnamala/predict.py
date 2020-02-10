from keras.models import load_model
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from basics import wav2mfcc, get_labels, label_num


##########################################################################
feature_dim_1 = 20
feature_dim_2 = 21
channel = 1
fs=16000
duration = 1  # seconds
############################################################################

##numpy.argmax(array/list) returns indices of the maximum value
def predict(wavfile, model):

    dict={}

    sample = wav2mfcc(wavfile)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)

    prediction = model.predict(sample_reshaped)#contains list of prediction
                                               #class values
    labels = get_labels()[0] # contains list of labels
    #print(labels)
    #print(prediction)
    print()#empty line
    print("the predicted classes values are:")
    print()#empty line
    for i in range(label_num):
        dict[labels[i]] = prediction[0][i] #converting to %

    print(dict)
    print()#empty line
    return labels[np.argmax(prediction)]

############################################################################


if __name__ == '__main__':

    name = 'model3_2.h5'
    model = load_model(name)
    keyboard=1
    while(keyboard==1):
        print ("Recording...")
        myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float32')
        sd.wait()
        print("finished recording")
        wav.write("monofile.wav", fs, myrecording)
        print()#empty line
        print()#empty line

        predictions = predict('monofile.wav', model=model)
        print("using model \"{}\"".format(name))
        print()#empty line
        if(predictions == 'noise'):
            print("Sorry,Can't hear your voice!!")
        else:
            print("We predicted you saying: \'{} \'".format(predictions))

        print()#empty line
        print("Enter 1 to try again:")
        keyboard = int(input())
