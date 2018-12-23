import pickle, pprint
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint


def load_data(file=''):
    pkl_file = open(file, 'rb')

    data = pickle.load(pkl_file)
    #pprint.pprint(data)

    pkl_file.close()
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]

    return((X_train, Y_train), (X_test, Y_test))

"""
load data
"""

(x_train, Y_train), (x_test, Y_test) = load_data(file='data/glucose_acc_set1_ws10_ss5_tl1_j5.pkl')

print("shape of x_train: ", np.shape(x_train))
print("shape of Y_train: ", np.shape(Y_train))
print("shape of x_test: ", np.shape(x_test))
print("shape of Y_test: ", np.shape(Y_test))


"""
define model
"""

hidden_size = 100
INPUT_SHAPE = (np.shape(x_train)[1], np.shape(x_train)[2])
target_length = 1

rnn = Sequential()
rnn.add(LSTM(hidden_size, return_sequences=False, input_shape=INPUT_SHAPE))
rnn.add(Dense(target_length, activation='linear'))

rnn.compile(loss='mean_squared_error', optimizer='adam')
rnn.summary()


"""
begin training
"""

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "rnn_model_v1.hdf5")

NUM_EPOCHS = 1000

checkpointer = ModelCheckpoint(filepath=RNN_FILE_MODEL, verbose=1, save_best_only=True)

rnn.fit(x=x_train,
        y=Y_train,
        epochs=NUM_EPOCHS,
        batch_size=1,
        validation_split=0.2,
        callbacks=[checkpointer]
        )
