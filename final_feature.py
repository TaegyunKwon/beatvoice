#-*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import librosa.display

BEAT_PATH = './output/'
data_path = './dataset/beat/'
FEATURE_PATH = './dataset/features/'
mfcc_path = 'mfcc'
dmfcc_path = 'dmfcc'
ddmfcc_path = 'ddmfcc'
mel_path = 'mel'
zcrsum_path = 'zcrsum'
sc_mat_path = 'sc_mat'
full_path = 'full'

MFCC_DIM = 30
MEL_DIM = 40
SR = 44100

drum_types = ["kick", "snare", "hi"]


class beat_feature(object):
  def __init__(self, time, y, sr):
    self.time = time
    self.y = y
    self.sr = sr

  def get_mfcc(self):
    self.mfcc, self.dmfcc, self.ddmfcc = extract_mfcc(self.y, self.sr)
    return self.mfcc, self.dmfcc, self.ddmfcc

  def get_mel(self):
    self.mel = extract_mel(self.y, self.sr)
    return self.mel

  def get_zero_crossing(self):
    self.zero_crossing = extract_zero_crossing(self,y, self.sr, self.mfcc.shape[1])
    return self.zero_crossing

  def get_spectral_centroid(self):
    sc_mat = librosa.feature.spectral_centroid(self.y, sr=self.sr)
    self.spectral_centroid = sc_mat

    return self.spectral_centroid

  def extract_features(self):
    try:
      self.mfcc
    except:
      self.get_mfcc()
    try:
      self.mel
    except:
      self.get_mel()
    try:
      self.zero_crossing
    except:
      self.get_zero_crossing()
    try:
      self.spectral_centroid
    except:
      self.get_spectral_centroid()

    features = np.concatenate([self.mfcc,
                               self.dmfcc,
                               self.ddmfcc,
                               self.mel,
                               self.zero_crossing,
                               self.spectral_centroid], axis=0)
    self.features = features
    return features


def save_feature(fn, path, feature):
    file_name = fn.replace('.wav', '.npy')
    save_file = os.path.join(FEATURE_PATH, path, file_name)

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, feature)


def extract_mfcc(y, sr):
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)
  dmfcc = librosa.feature.delta(mfcc)
  ddmfcc = librosa.feature.delta(mfcc, order=2)
  return mfcc, dmfcc, ddmfcc


def extract_mel(y, sr):
  S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)
  D = np.abs(S) ** 2
  mel_basis = librosa.filters.mel(sr, 1024, n_mels=MEL_DIM)
  mel_S = np.dot(mel_basis, D)
  log_mel_S = librosa.power_to_db(mel_S)
  return log_mel_S


def extract_zero_crossing(y, sr, repeat_length):
  zero_crossings = librosa.zero_crossings(y, pad=False)
  zcrsum = sum(zero_crossings)
  zcrsum = np.repeat(zcrsum, repeat_length)
  zcrsum = np.reshape(zcrsum, (-1, 1))
  zcrsum = zcrsum.T

  return zcrsum


def process_beat(y, sr, t=None):
  # beat_feature(t, y, sr)

  mfcc, dmfcc, ddmfcc = extract_mfcc(y, sr)

  mel = extract_mel(y, sr)

  zcrsum = extract_zero_crossing(y, sr, mfcc.shape[1])

  sc_mat = librosa.feature.spectral_centroid(y, sr=sr)

  full_feature = np.concatenate([mfcc,
                                 dmfcc,
                                 ddmfcc,
                                 mel,
                                 zcrsum,
                                 sc_mat], axis=0)

  return full_feature


if __name__ == '__main__':
  for drum_name in drum_types:

      temp_file_list = os.listdir(BEAT_PATH + drum_name)
      file_list = [el for el in temp_file_list if el.endswith('.wav')]

      c=0
      #random_tvt=np.random.choice(range(howmany), howmany, replace=False)        #shuffle

      #for name in random_tvt:        #shuffle
      for file_name in file_list:
          c = c + 1
          if not (c % 10):
              print(drum_name, c)

          y, sr = librosa.load(os.path.join(BEAT_PATH, drum_name, file_name), sr=SR)

          full_feature = process_beat(y, sr)
          save_feature(drum_name + "_" + file_name, full_path, full_feature)
