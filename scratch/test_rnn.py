import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_MODEL = os.path.join(DIR_PATH, "rnn_models")
RNN_FILE_MODEL = os.path.join(DIR_MODEL, "rnn_model_v1.hdf5")

def load_data(file='data/glucose_acc_ws10_ss5.pkl'):
    pkl_file = open(file, 'rb')

    data = pickle.load(pkl_file)
    #pprint.pprint(data)

    pkl_file.close()
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]

    return((X_train, Y_train), (X_test, Y_test))

"""reshape in and outputs"""

(X_train, Y_train), (X_test, Y_test) = load_data('data/glucose_acc_set1_ws10_ss1_tl1_j5.pkl')
print("x_train shape: " ,X_train.shape)
print("y_train shape: " ,Y_train.shape)
print("x_test shape: " ,X_test.shape)
print("y_test shape: " ,Y_test.shape)

"""load model"""
rnn = load_model(RNN_FILE_MODEL)
rnn.load_weights(RNN_FILE_MODEL)

"""check results"""

predictions = rnn.predict(X_train)

print("prediction shape: ", np.shape(predictions))
print("Y_test shape: ", np.shape(Y_train))

Y_train = np.reshape(Y_train, (len(Y_train), 1))

abs_diff = predictions - Y_train
print("absolute differences: ", abs_diff)
print("relative differences: ", (predictions - Y_train)/Y_train )



a = 800
b = 1200
plt.plot(abs_diff[a:b], label='differences')
plt.plot(Y_train[a:b], 'r', label='glucose values')
plt.plot(predictions[a:b], 'g', label='predictions')
plt.legend()
plt.savefig('plots/{}_{}.png'.format(a,b))
plt.show()



"""
i = 5
plt.plot(np.append(np.zeros(10), abs_diff[i]), label='difference')
plt.plot(np.append(Y_train[i], Y_test[i]), label='glucose_values_hist')
#plt.plot(Y_test[0], label='glucose_values_real')
plt.plot(np.append(Y_train[i], predictions[i]), label='prediction')
plt.legend()
plt.savefig('plots/i_{}'.format(i))
plt.show()
"""
