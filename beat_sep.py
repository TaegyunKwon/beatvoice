#-*- coding: utf-8 -*-
from __future__ import division

import librosa
import madmom
import numpy as np
import utils
import constants


def pattern_cut(file_path, times, buffer=constants.BUFFER, verbose=True):
    y, sr = librosa.load(file_path, sr=constants.SR)
    beat_time = librosa.time_to_samples(times, sr=constants.SR)
    # 8 patterns
    pattern_samples = []
    for n in range(8):
        pattern_sample = y[beat_time[2*n]: beat_time[2*n+1]]
        name = 'p{:d}'.format(n)
        first_onset = librosa.onset.onset_detect(y=pattern_sample, sr=constants.SR, units='samples')[0]
        # cut_range = (first_onset - int(win_pre*SR), first_onset + int(win_post*SR))
        pattern_sample = pattern_sample[first_onset - buffer:]
        if verbose:
            print('p{:d}, range: {:0.2f} ~ {:0.2f}'.format(n, (beat_time[2*n] + first_onset - buffer)/constants.SR, beat_time[2*n+1]/constants.SR))
            # _, tail = utils.split_head_and_tail(file_path)
            # librosa.output.write_wav('{}_{}.wav'.format(tail, name), pattern_sample, sr)
        librosa.output.write_wav('{}.wav'.format(name), pattern_sample, sr)
        pattern_samples.append(pattern_sample)

    return pattern_samples


def beat_comb_peakpick(file_path):
    """
    Comb filter based beat-tracking method.
    because basic function may gives duplicate(too close) or disordered beats, we added uniqueness / 0.2 sec threshold.

    :param file_path: path of audio file
    :return: list of beat locations, in sec unit.

    REFERENCE
    http://madmom.readthedocs.io/en/latest/modules/features/beats.html#madmom.features.beats.BeatTrackingProcessor
    """
    beat_est = madmom.features.onsets.RNNOnsetProcessor()(file_path)
    beat_track_processor = madmom.features.beats.BeatTrackingProcessor(fps=100)

    loc_beats = beat_track_processor(beat_est)
    # plt.plot(loc_beats)

    loc_beats = np.unique(loc_beats)
    m = 1
    loc_beats_new = np.zeros(loc_beats.shape)
    last_beat = loc_beats[0]
    loc_beats_new[0] = loc_beats[0]
    for n in range(1, len(loc_beats)):
        if loc_beats[n] - last_beat >= constants.LEAST_GAP:
            last_beat = loc_beats[n]
            loc_beats_new[m] = last_beat
            m += 1
    loc_beats = loc_beats_new[:m]

    return loc_beats #beat


def get_beat_sample_path(label, file_name, n_pattern, n_beat):
    if label == 0:
        drum_type = 'kick'
    elif label == 1:
        drum_type = 'hi'
    elif label == 2:
        drum_type = 'snare'
    return './output/{}/{}_p{}_beat{}.wav'.format(drum_type, file_name, n_pattern, n_beat)


def sample_cut(file_path, pre_adjust, post_adjust=None, verbose=False):
    utils.maybe_make_dir('./output/kick')
    utils.maybe_make_dir('./output/hi')
    utils.maybe_make_dir('./output/snare')

    pre_adjust = pre_adjust
    _, file_name = utils.split_head_and_tail(file_path)
    file_name = file_name.replace('.wav', '')
    if post_adjust is None:
        post_adjust = pre_adjust

    times = [11.5 + pre_adjust, 18.5 + post_adjust
        , 25.3 + pre_adjust, 32.2 + post_adjust
        , 39.3 + pre_adjust, 46.2 + post_adjust
        , 53 + pre_adjust, 60 + post_adjust
        , 66.5 + pre_adjust, 73.7 + post_adjust
        , 80.5 + pre_adjust, 87.5 + post_adjust
        , 94.2 + pre_adjust, 101.2 + post_adjust
        , 108 + pre_adjust, 115 + post_adjust]  # sec

    _ = pattern_cut(file_path, times)

    beats = []
    labels = []
    for n in range(8):
        name = 'p{:d}.wav'.format(n)
        y, sr = librosa.core.load(name, sr=constants.SR)
        beats_index = (librosa.time_to_samples(beat_comb_peakpick(name), sr=constants.SR))
        n_beats = len(beats_index)
        if verbose:
            print('{}_p{:d}_#beat={:d}'.format(file_name, n, n_beats))
        if n_beats < 10:
            print('Critical: {}_p{:d}_#beat={:d}'.format(file_name, n, n_beats))
            continue
        elif n_beats < 16:
            # print('Warning: {}_p{:d}_#beat={:d}'.format(file_name, n, n_beats))
            print('Critical: {}_p{:d}_#beat={:d}'.format(file_name, n, n_beats))
            continue

        for m in range(n_beats):
            if m >= 16:
                break

            beat_index = beats_index[m]
            out = y[int(max(0, beat_index - constants.PRE_ONSET*constants.SR)): int(beat_index + constants.POST_ONSET*constants.SR)]
            if m == 0:
                diff = beat_index - constants.BUFFER
                if m == 0 and abs(diff) > 0.1 * constants.SR:
                    print('Large onset diff:{:d}ms {}, p{:d}, beat{:d}'.format(int(diff / constants.SR * 1e3), file_name, n, m))
            beats.append(y)

            label = constants.LABEL[n, m]
            librosa.output.write_wav(get_beat_sample_path(label, file_name, n, m), out, sr)
            labels.append(label)

            if verbose:
                pass
                # librosa.output.write_wav('./output/'+str(p)+'/'+str(beat_number)+'_'+str(label[p,beat_n])+'.wav', out, sr)

    return beats, labels


if __name__ == '__main__':
    sample_cut('./Samples/sangeun.wav', 0)
    sample_cut('./Samples/choi3.wav', 0.4)
    sample_cut('./Samples/joong.wav', 0.5)

    sample_cut('./Samples/wonil.wav', 0.1)
    sample_cut('./Samples/choi4.wav', 0)
    sample_cut('./Samples/jung.wav', 0.1)

    sample_cut('./Samples/wonjun_n.wav', 2.7)
    sample_cut('./Samples/dongju.wav', 0.1)

    sample_cut('./Samples/kiki.wav', 0.4)
    sample_cut('./Samples/sanggue.wav', 0.3)
    sample_cut('./Samples/noname_n.wav', -0.05)
    sample_cut('./Samples/choi1.wav', 0.4)
    sample_cut('./Samples/daenam.wav', -0.1)
    sample_cut('./Samples/jinhong.wav', -0.4)

    sample_cut('./Samples/MS_data_1.wav', 0.4)
    sample_cut('./Samples/MS_data_2_n.wav', -1.2)
    sample_cut('./Samples/MS_data_3.wav', -0.3)
    sample_cut('./Samples/MS_data_4.wav', -0.3)
    sample_cut('./Samples/MS_data_6.wav', -0.2)
    sample_cut('./Samples/MS_data_7.wav', -0.8)

    # sample_cut('./Samples/woojin.wav',0.5)
    # sample_cut('./Samples/choi6.wav',0.3)
    # sample_cut('./Samples/choi2.wav',0.2)
    # out=sample_cut('./Samples/choi5.wav',-0.6,out[0],out[1],out[2])
    # sample_cut('./Samples/tae.wav',0.0)#check 1,6
