#calls audioaugment.py and send the audio files path arguments
import os

path = "datasets/"

def get_labels(path=path):
    labels = os.listdir(path)
    return labels


if __name__ == "__main__":

    labels= get_labels(path)
    print(labels)
    for label in labels:
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            os.system("/media/logic/Workplace/projects/majorProject/nepaliBarnamala/audioaugment.py 4 {}".format(wavfile))
