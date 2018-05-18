import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from labels import new_labels

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load("heartbeat-sounds/set_a/"+fp)
        raw_sounds.append(X)
        #raw_sounds.append(sr)
    return raw_sounds


def extract_feature(file_name):
    #print(file_name)
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,new_labels, file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)

    heartbeats = {"artifact" :0 ,"normal":1,"extrahls":1, "murmur":2}

    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):

            filename = fn.split('/')[2]

            label = new_labels[filename]

            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


parent_dir = "heartbeat-sounds"
tr_sub_dirs = ["set_a_training"]
ts_sub_dirs = ["set_a_testing"]

def classify():

    tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs, new_labels)
    ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs, new_labels)
    #print(tr_labels)
    #print ('-----------------------------')
    tr_labels = one_hot_encode(tr_labels)
    ts_labels = one_hot_encode(ts_labels)
    #print(tr_features[0])

    #print(tr_labels)


    training_epochs = 500
    n_dim = tr_features.shape[1]
    n_classes = 3
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.01

    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,n_classes])

    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two],mean = 0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

    #cost_function = -tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1 - y_))
    cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
    #cost_function = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    #cost_function = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) )
    #optimizer = tf.train.AdamOptimizer().minimize(cost_function)

    init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost_history = np.empty(shape=[1], dtype=float)
    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            _, cost = sess.run([optimizer, cost_function], feed_dict={X: tr_features, Y: tr_labels})
            cost_history = np.append(cost_history, cost)
            print("Epoch "+str(epoch)+"/"+str(training_epochs)+", Cost: "+str(cost))

        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: ts_features})
        y_true = sess.run(tf.argmax(ts_labels, 1))
        print("Test accuracy: ", round(sess.run(accuracy,
                                                   feed_dict={X: ts_features, Y: ts_labels}), 3))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(cost_history)
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    plt.show()

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    print("F-Score:", round(f, 3))

classify()
