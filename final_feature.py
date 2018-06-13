#-*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import librosa.display
import constants


data_path = './dataset/beat/'

mfcc_path = 'mfcc'
dmfcc_path = 'dmfcc'
ddmfcc_path = 'ddmfcc'
mel_path = 'mel'
zcrsum_path = 'zcrsum'
sc_mat_path = 'sc_mat'
full_path = 'full'
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

  def extract_features(self, feature_list=['mfcc', 'mel', 'zero_crossing', 'centroid']):
    features = process_beat(self.y, self.sr, feature_list=feature_list)
    self.features = features
    return self.features


def save_feature(fn, path, feature):
    file_name = fn.replace('.wav', '.npy')
    save_file = os.path.join(constants.FEATURE_PATH, path, file_name)

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, feature)


def extract_mfcc(y, sr):
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=constants.MFCC_DIM)
  dmfcc = librosa.feature.delta(mfcc)
  ddmfcc = librosa.feature.delta(mfcc, order=2)
  return mfcc, dmfcc, ddmfcc


def extract_mel(y, sr):
  S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)
  D = np.abs(S) ** 2
  mel_basis = librosa.filters.mel(sr, 1024, n_mels=constants.MEL_DIM)
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


def process_beat(y, sr, feature_list=['mfcc', 'mel', 'zero_crossing', 'centroid']):
  # TODO: better to do it with dicts.
  full_feature = []
  if 'mfcc' in feature_list:
    mfcc, dmfcc, ddmfcc = extract_mfcc(y, sr)
    full_feature.append(mfcc)
    full_feature.append(dmfcc)
    full_feature.append(ddmfcc)
  if 'mel' in feature_list:
    mel = extract_mel(y, sr)
    full_feature.append(mel)
  if 'zero_crossing' in feature_list:
    zcrsum = extract_zero_crossing(y, sr, mfcc.shape[1])
    full_feature.append(zcrsum)
  if 'centroid' in feature_list:
    sc_mat = librosa.feature.spectral_centroid(y, sr=sr)
    full_feature.append(sc_mat)

  full_feature = np.concatenate([full_feature], axis=0)

  return full_feature


if __name__ == '__main__':
  for drum_name in drum_types:

      temp_file_list = os.listdir(constants.BEAT_PATH + drum_name)
      file_list = [el for el in temp_file_list if el.endswith('.wav')]

      c=0
      for file_name in file_list:
          c = c + 1
          if not (c % 10):
              print(drum_name, c)

          y, sr = librosa.load(os.path.join(constants.BEAT_PATH, drum_name, file_name), sr=constants.SR)

          full_feature = process_beat(y, sr)
          save_feature(drum_name + "_" + file_name, full_path, full_feature)
