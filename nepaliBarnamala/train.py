import keras
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM
from basics import save_data_to_array, get_train_test, label_num


###############################################################################
# Feature dimension
feature_dim_1 = 20 #dependes on size of mfcc
feature_dim_2 = 21 #depends on size of mfcc
channel = 1
epochs = 300
batch_size = 100
verbose = 2


##################### CNN ARCHITECTURE ########################################

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2),
                    activation='relu',
                    input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))#activated only in training phase
    model.add(Flatten())
    model.add(Dense(128, activation='relu', use_bias = True ))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu', use_bias = True))
    model.add(Dropout(0.4))
    model.add(Dense(label_num, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model



###############################################################################

if __name__ == '__main__':


    #save_data_to_array() #activate only for first training to generate .npy files
    X_train, X_test, y_train, y_test = get_train_test()
    #X_train contains datasets for training
    #y_train contains label index for corresponding training vectors

    #X_test contains datasets for testing
    #y_test contains label index for corresponding testing vectors



    # Reshaping to perform 2D convolution
    X_train = X_train.reshape(X_train.shape[0],
                              feature_dim_1,
                              feature_dim_2,
                              channel)
    X_test = X_test.reshape(X_test.shape[0],
                            feature_dim_1,
                            feature_dim_2,
                            channel)

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    #creating tensorboard logger
    logger = TensorBoard(
              log_dir = 'logs/model4_aug',
              histogram_freq = 5,
              write_graph = True)

    model = get_model()
    model.fit(X_train,
              y_train_hot,
              batch_size=batch_size,
              epochs=epochs,
              shuffle =  True,
              verbose=verbose,
              validation_data=(X_test, y_test_hot),
              callbacks = [logger]
             )
    #test_error_rate = model.evaluate (X_test, y_test_hot, verbose = 0)
    #print("MSE is {}".format(test_error_rate))
    model.save('model4_aug.h5')
