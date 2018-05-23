import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from scipy.io import wavfile
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import decimate
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.utils import np_utils
from keras.regularizers import l2
import labels
import wavelet_transformation

directory = 'set_'+sys.argv[1]
wavelet_transform = int(sys.argv[2])

INPUT_LIB = 'heartbeat-sounds/'
SAMPLE_RATE = 4000
CSV = directory+'.csv'
DATASET = directory+'/'

new_labels,label_array = labels.read_labels(CSV)

def clean_filename(fname, string):
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':
        file_name = string + file_name
    return file_name

def load_wav_file(name, path, wavelet_transform=1):
    _, audio_data = wavfile.read(path + name)
    #print(_)
    #assert _ == SAMPLE_RATE
    if wavelet_transform == 1:
        audio_data = wavelet_transformation.wavelet_transformation(audio_data)
    return audio_data

def repeat_to_length(arr, length):
    """Repeats the numpy 1D array to given length, and makes datatype float"""
    result = np.empty((length, ), dtype = 'float32')
    l = len(arr)
    pos = 0
    while pos + l <= length:
        result[pos:pos+l] = arr
        pos += l
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result


def get_me_my_model():
    model = Sequential()
    model.add(Conv1D(filters=4, kernel_size=9, activation='relu',
                    input_shape = x_train.shape[1:],
                    kernel_regularizer = l2(0.025)))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=9, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))
    model.add(GlobalAvgPool1D())
    model.add(Dense(3, activation='softmax'))
    return model

df = pd.read_csv(INPUT_LIB + CSV)
df['fname'] = df['fname'].apply(clean_filename, string='Aunlabelledtest')
df['label'].fillna('unclassified')
df['time_series'] = df['fname'].apply(load_wav_file, path=INPUT_LIB +  DATASET, wavelet_transform=wavelet_transform)
df['len_series'] = df['time_series'].apply(len)
MAX_LEN = max(df['len_series'])
df['time_series'] = df['time_series'].apply(repeat_to_length, length=MAX_LEN)

x_data = np.stack(df['time_series'].values, axis=0)

new_labels = np.array(label_array, dtype='int')
y_data = np_utils.to_categorical(new_labels)


x_train, x_test, y_train, y_test, train_filenames, test_filenames = \
    train_test_split(x_data, y_data, df['fname'].values, test_size=0.3)

#sys.exit()

x_train = decimate(x_train, 8, axis=1, zero_phase=True)
x_train = decimate(x_train, 8, axis=1, zero_phase=True)
x_train = decimate(x_train, 4, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 4, axis=1, zero_phase=True)

#Scale each observation to unit variance, it should already have mean close to zero.
x_train = x_train / np.std(x_train, axis=1).reshape(-1,1)
x_test = x_test / np.std(x_test, axis=1).reshape(-1,1)

x_train = x_train[:,:,np.newaxis]
x_test = x_test[:,:,np.newaxis]

model = get_me_my_model()

def batch_generator(x_train, y_train, batch_size):
    """
    Rotates the time series randomly in time
    """
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    full_idx = range(x_train.shape[0])

    while True:
        batch_idx = np.random.choice(full_idx, batch_size)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis=0)

        yield x_batch, y_batch

weight_saver = ModelCheckpoint('set_a_weights.h5', monitor='val_loss',
                               save_best_only=True, save_weights_only=True)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8**x)

hist = model.fit_generator(batch_generator(x_train, y_train, 8),
                   epochs=30, steps_per_epoch=100,
                   validation_data=(x_test, y_test),
                   callbacks=[weight_saver, annealer],
                   verbose=2)

model.load_weights('set_a_weights.h5')

#for i in range(len(model.metrics_names)):
#print(str(model.metrics))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_xlabel('Epochs')
ax.set_ylabel('Error')
plt.plot(hist.history['loss'], color='b', label="Training")
plt.plot(hist.history['val_loss'], color='r', label="Cross Validation")
plt.legend(loc="best")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
plt.plot(hist.history['acc'], color='b', label="Training")
plt.plot(hist.history['val_acc'], color='r', label="Cross Validation")
plt.legend(loc="best")
plt.show()

