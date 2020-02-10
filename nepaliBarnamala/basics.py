import os
import librosa
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

############################################################################
DATA_PATH = "/media/logic/Workplace/majorProject/majorProject_files/nepaliBarnamala/aug_datasets/"
##############################################################################



num_classes = len(os.listdir(DATA_PATH))#defining the classes to classify the
                                        #training data or number of labels
#print(num_classes)
label_num = num_classes

########################## Generate Image from MFCC ###########################

def make_image(data, label, size=(1, 1), dpi=200):
    newpath="/media/logic/Workplace/projects/majorProject/nepaliBarnamala/image"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath1="/media/logic/Workplace/projects/majorProject/nepaliBarnamala/image/{}".format(label)
    if not os.path.exists(newpath1):
        os.makedirs(newpath1)
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('coolwarm')
    ax.imshow(data, aspect='equal')
    os.chdir(newpath1)
    plt.savefig(label+str(random.randint(0, 50))+random.choice('abcdefghijklmnopqrstuvwxyz')+str(random.randint(0, 100))+".png", dpi=dpi)



########################## MFCC CALCULATION ###############################
def wav2mfcc(file, max_len=21):#change max_len
    wave, sr = librosa.load(file, mono=True, sr=None)
    wave = wave[::3]#applying interval to make it discrete time signal
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    #  making  the size of mfcc[20][11]
    # use fixed size for every labels mfcc you can alter it to see the change


    # If maximum length exceeds mfcc lengths then pad the remaining ones
    # that is adding zeros to make it's size [20][11]
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


################################### get labels from dataset #####################
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))#gives an array containing numbers
                                                # from 0 to len-1
    #print(labels,label_indices, to_categorical(label_indices))
    return labels, label_indices, to_categorical(label_indices)
    #get_labels()
    #to_categorical converts the given decimal to the binary system

########################### Save MFCC Vectors in .npy file ###################
def save_data_to_array(path=DATA_PATH, max_len=21):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' +
                    wavfile for wavfile in os.listdir(path + '/' + label)]
        #print(wavfiles)
        for wavfile in tqdm(wavfiles,
                            "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len = max_len)
            #make_image(mfcc, label)#generate image from mfcc
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

############################### get training and test data #############################
def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')#loads labels[0].npy files
    y = np.zeros(X.shape[0])#creates an array filled with X.shape[0] numbers of
                            # zeros

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))#stacks x with X  vertically  (row wise)
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)
    # y is array containing the corresponding label index number for
    #each values in X

    return train_test_split(X, y,
                            test_size= (1 - split_ratio),
                            random_state=random_state,
                            shuffle=True)
#splits datasets(X) into 2 parts first part  for training, second part for test and
#also return two other arrays indicating the first and second parts labels(y)
#test size is defined in the function as 1-split_ratio
