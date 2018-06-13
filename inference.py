from __future__ import division

import librosa
import numpy as np
import pickle
from sklearn.externals import joblib
import pretty_midi
from final_feature import BeatFeature
from final_train_test import normalize_features_object, normalize_features, mean_feature
import argparse
import constants


def pick_onset(audio_file, onset_strength_th=4, onset_time_th=0.12):
  y, sr = librosa.load(audio_file)
  onset_samples = librosa.onset.onset_detect(y, sr=constants.SR, units='samples')
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


def infer_to_midi(audio_file, model, scaler, out_name=None):
  onset_post, y, sr = pick_onset(audio_file)
  features = []
  for el in onset_post:
    window_start = max(0, int(el-0.1*sr))
    window_end = min(len(y), int(el+0.3*sr))
    feature = BeatFeature(el/sr, y[window_start: window_end], sr)
    feature.get_full_features()
    features.append(feature)

  midi = pretty_midi.PrettyMIDI(initial_tempo=120)
  inst = pretty_midi.Instrument(program=119, is_drum=True, name='BeatVoice')
  midi.instruments.append(inst)

  for el in features:
    el.full_feature = mean_feature(el.full_feature)
  normalize_features_object(features)
  # features = normalize_features_object(features)
  for el in features:
    feature = el.full_feature
    feature = scaler.transform(feature.reshape(1,-1))
    prediction = int(model.predict(feature))

    if prediction == 1:
      inst.notes.append(pretty_midi.Note(80, 36, el.time, el.time+0.2))
    elif prediction == 3:
      inst.notes.append(pretty_midi.Note(80, 42, el.time, el.time + 0.2))
    elif prediction == 2:
      inst.notes.append(pretty_midi.Note(80, 40, el.time, el.time + 0.2))
  if out_name is None:
    out_name = infer_file.replace('.wav', '.mid')
  midi.write(out_name)
  return midi


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--infer_file')
  parser.add_argument('--scaler', default='stat/scaler.save')
  parser.add_argument('--model', default='test_model.sav')
  parser.add_argument('--output_file', default=None)
  args = parser.parse_args()

  infer_file = args.infer_file
  model = args.model
  scaler = args.scaler

  onset_post, y, sr = pick_onset(infer_file)

  model = pickle.load(open(model, 'rb'))
  scaler = joblib.load(scaler)
  infer_to_midi(infer_file, model, scaler, out_name=args.output_file)
