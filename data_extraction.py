import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import librosa
import labels
import wavelet_transformation


def extract_feature(file_name, wavelet_transform):
    audio_data, sample_rate = librosa.load(file_name)

    if wavelet_transform == 1:
        audio_data = wavelet_transformation.wavelet_transformation(audio_data)

    stft = np.abs(librosa.stft(audio_data))
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(audio_data, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,new_labels, wavelet_transform, file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)

    i = 1

    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):

            filename = fn.split('/')[2]

            label = new_labels[filename]

            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn, wavelet_transform)
            except Exception as e:
              print("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)

            print(str(i) + " files parsed")
            i+=1
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def get_features_labels(data_set,wavelet_transform=0):

    parent_dir = 'heartbeat-sounds'
    training_set = data_set+'_training'
    testing_set = data_set+'_testing'
    CSV = data_set+'.csv'

    if wavelet_transform == 1:
        print('Performing Wavelet Transformation')

    tr_sub_dirs = [training_set]
    ts_sub_dirs = [testing_set]
    new_labels, label_array = labels.read_labels(CSV)

    tr_features, tr_labels = parse_audio_files(parent_dir, tr_sub_dirs, new_labels, wavelet_transform)
    ts_features, ts_labels = parse_audio_files(parent_dir, ts_sub_dirs, new_labels, wavelet_transform)

    tr_labels = one_hot_encode(tr_labels)
    ts_labels = one_hot_encode(ts_labels)

    return tr_features, tr_labels, ts_features, ts_labels