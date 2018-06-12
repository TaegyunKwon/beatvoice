from __future__ import division

import librosa
import numpy as np
import pickle
from sklearn.externals import joblib
import pretty_midi
from final_feature import beat_feature

def pick_onset(audio_file, onset_strength_th=4, onset_time_th=0.12):
  y, sr = librosa.load(audio_file)
  onset_samples = librosa.onset.onset_detect(y, sr=sr, units='samples')
  onset_strength = librosa.onset.onset_strength(y, sr)
  onset_post = []
  for el in onset_samples:
    if onset_strength[el//512] >= onset_strength_th:
      onset_post.append(el)
  while True:
    rerun = False
    for n in range(len(onset_post)-1):
      current_onset = onset_post[n]
      post_onset = onset_post[n+1]
      if post_onset - current_onset <= sr*onset_time_th:
        rerun = True
        if onset_strength[post_onset//512] > onset_strength[current_onset // 512]:
          del onset_post[n]
        else:
          del onset_post[n+1]
        break
    if rerun:
      pass
    else:
      break
  return onset_post, y, sr


def infer_to_midi(audio_file, model, scaler):
  onset_post, y, sr = pick_onset(audio_file)
  features = []
  for el in onset_post:
    window_start = max(0, int(el-0.1*sr))
    window_end = min(len(y), int(el+0.3*sr))
    feature = beat_feature(el/sr, y[window_start: window_end], sr)
    feature.extract_features()
    features.append(feature)

  midi = pretty_midi.PrettyMIDI(initial_tempo=120)
  inst = pretty_midi.Instrument(program=119, is_drum=True, name='BeatVoice')
  midi.instruments.append(inst)

  mfcc_time = 'off'
  frame_n = 3

  mfcc = np.zeros(shape=frame_n)

  for el in features:
    feature = el.features
    feature_mean = np.mean(feature, axis=1)

    if mfcc_time == 'on':
      w = int(feature[0].shape[0] * (10.0 / float(frame_n)) * 0.1)
      s = 0
      e = w
      for m in range(0, frame_n):
        mfcc[m] = np.mean(feature[1:30, s:e])
        s = s + w
        e = e + w
        if m == frame_n - 2:
          e = feature[0].shape[0]

      feature=np.concatenate((feature_mean,mfcc))

    else:
      feature = feature_mean



    feature = scaler.transform(feature.reshape(1,-1))
    prediction = int(model.predict(feature))

    if prediction == 1:
      inst.notes.append(pretty_midi.Note(80, 36, el.time, el.time+0.2))
    elif prediction == 3:
      inst.notes.append(pretty_midi.Note(80, 42, el.time, el.time + 0.2))
    elif prediction == 2:
      inst.notes.append(pretty_midi.Note(80, 40, el.time, el.time + 0.2))
  midi.write(TEST_FILE.replace('.wav', '.mid'))
  return midi


if __name__ == '__main__':
  TEST_FILE = 'Samples/test/wonil.wav'
  MODEL = 'finalized_model.sav'
  SCALER = 'stat/scaler.save'

  onset_post, y, sr = pick_onset(TEST_FILE)

  model = pickle.load(open(MODEL, 'rb'))
  scaler = joblib.load(SCALER)
  infer_to_midi(TEST_FILE, model, scaler)
