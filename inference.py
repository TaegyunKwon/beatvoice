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
import madmom

VEL_MIN = 80
VEL_MAX = 120


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

def pick_onset_madmom(audio_file, onset_strenght_pre=8, onset_strenght_post=3):
  y, sr = librosa.load(audio_file, 44100)
  onset_samples = librosa.onset.onset_detect(y, sr, units='samples')
  onset_strength = librosa.onset.onset_strength(y, sr)
  onset_post = []
  for el in onset_samples:
    if onset_strength[el//512] >= onset_strenght_pre:
      onset_post.append(el)

  y_onset=y[onset_post[0]-3000:]

  beat_est = madmom.features.onsets.RNNOnsetProcessor()(y_onset)

  onset_post = []
  ycut_onset_strength = librosa.onset.onset_strength(y_onset, sr)

  beat_track_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)
  loc_beats = beat_track_processor(beat_est)
  onset_temp=librosa.time_to_samples(loc_beats,sr)

  for onset_temp_samples in onset_temp:
      if ycut_onset_strength[onset_temp_samples // 512] >= onset_strenght_post:
          onset_post.append(onset_temp_samples)
  return onset_post, y_onset, sr



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

  rms = []
  for el in features:
    rms.append(np.max(np.log(librosa.feature.rmse(el.y))))
  max_rms = max(rms)
  min_rms = min(rms)

  for el in features:
    rms = np.max(np.log(librosa.feature.rmse(el.y)))
    rel_rms = abs((rms-min_rms)/(max_rms-min_rms))
    print rel_rms
    vel = int(rel_rms*45 + 80)

    feature = el.full_feature
    feature = scaler.transform(feature.reshape(1,-1))
    prediction = int(model.predict(feature))

    if prediction == 1:
      inst.notes.append(pretty_midi.Note(vel, 36, el.time, el.time+0.2))
    elif prediction == 3:
      inst.notes.append(pretty_midi.Note(vel, 42, el.time, el.time + 0.2))
    elif prediction == 2:
      inst.notes.append(pretty_midi.Note(vel, 40, el.time, el.time + 0.2))
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

  # onset_post, y, sr = pick_onset(infer_file)
  onset_post, y, sr = pick_onset_madmom(infer_file)

  model = pickle.load(open(model, 'rb'))
  scaler = joblib.load(scaler)
  infer_to_midi(infer_file, model, scaler, out_name=args.output_file)
