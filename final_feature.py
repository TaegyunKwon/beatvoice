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
drum_types = ["kick", "snare", "hi"]


class BeatFeature(object):
  def __init__(self, time, y, sr):
    self.time = time
    self.y = y
    self.sr = sr
    self.features ={}

  def get_mfcc(self):
    mfcc, dmfcc, ddmfcc = extract_mfcc(self.y, self.sr)
    self.features['mfcc'] = mfcc
    self.features['dmfcc'] = dmfcc
    self.features['ddmfcc'] = ddmfcc

    return mfcc, dmfcc, ddmfcc

  def get_mel(self):
    mel = extract_mel(self.y, self.sr)
    self.features['mel'] = mel
    return mel

  def get_zero_crossing(self):
    try:
      zero_crossing = extract_zero_crossing(self.y, self.sr, self.features['mfcc'].shape[1])
    except:
      self.get_mfcc()
      zero_crossing = extract_zero_crossing(self.y, self.sr, self.features['mfcc'].shape[1])
    self.features['zero_crossing'] = zero_crossing
    return zero_crossing

  def get_spectral_centroid(self):
    sc_mat = librosa.feature.spectral_centroid(self.y, sr=self.sr, hop_length=constants.HOP)
    self.features['centroid'] = sc_mat
    return sc_mat

  def get_full_features(self, key_list=['mfcc', 'dmfcc', 'ddmfcc', 'mel', 'zero_crossing', 'centroid']):
    feature_list = []
    for key in key_list:
      try:
        self.features[key].value
      except:
        if key in ['mfcc', 'dmfcc', 'ddmfcc']:
          self.get_mfcc()
        elif key in ['mel']:
          self.get_mel()
        elif key in ['zero_crossing']:
          self.get_zero_crossing()
        elif key in ['centroid']:
          self.get_spectral_centroid()
      feature_list.append(self.features[key])

    full_feature = np.concatenate(feature_list, axis=0)
    self.full_feature = full_feature
    return full_feature


def save_feature(fn, feature):
    file_name = fn.replace('.wav', '.npy')
    save_file = os.path.join(constants.FEATURE_PATH, file_name)

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    np.save(save_file, feature)


def extract_mfcc(y, sr):
  mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr,
                                                         n_fft=constants.FFT_LEN,
                                                         hop_length=constants.HOP,
                                                         ))
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=constants.MFCC_DIM, S=mel)
  dmfcc = librosa.feature.delta(mfcc)
  ddmfcc = librosa.feature.delta(mfcc, order=2)
  return mfcc, dmfcc, ddmfcc


def extract_mel(y, sr):
  S = librosa.core.stft(y, n_fft=constants.FFT_LEN, hop_length=constants.HOP, win_length=constants.FFT_LEN)
  D = np.abs(S) ** 2
  mel_basis = librosa.filters.mel(sr, constants.FFT_LEN, n_mels=constants.MEL_DIM)
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

          beat_feature = BeatFeature(None, y, sr)
          full_feature = beat_feature.get_full_features()
          save_feature(drum_name + "_" + file_name, full_feature)
