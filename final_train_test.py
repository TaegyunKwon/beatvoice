#-*- coding: utf-8 -*-
import sys
import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from final_feature_summary import *
import librosa.display
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import pickle
import random
from sklearn.externals import joblib
import argparse
from sklearn.neural_network import MLPClassifier
import constants


random.seed(634)


def parse_name(feature_name):
    split = feature_name.split('_')
    drum_type = split[0]
    if drum_type == 'kick':
        label = 1
    elif drum_type == "snare":
        label = 2
    elif drum_type == "hi":
        label = 3

    beat_id = int(split[-1].replace('.npy', '').replace('beat', ''))
    pattern_id = int(split[-2].replace('p', ''))

    id = '_'.join(split[1: -2])

    return id, label, beat_id, pattern_id


def load_data_feature():
    file_list = os.listdir(constants.FEATURE_PATH)
    feature_list = []
    pattern_list = []
    id_list = []
    for el in file_list:
        id, label, beat_id, pattern_id = parse_name(el)
        feature_data = np.load(os.path.join(constants.FEATURE_PATH, el))
        feature_list.append((feature_data, id, label, beat_id, pattern_id))
        pattern_list.append('{}_p{}'.format(id, pattern_id))
        id_list.append(id)

    pattern_list = list(set(pattern_list))
    id_list = list(set(id_list))

    return feature_list, pattern_list, id_list


def make_dataset(data_unit='ID', train_ratio=0.6, valid_ratio=0.2):
    feature_list, pattern_list, id_list = load_data_feature()

    if data_unit == 'Note':
        random.shuffle(feature_list)
        n_total = len(feature_list)
        train_X = [el[0] for el in feature_list[:int(n_total*train_ratio)]]
        train_Y = [el[2] for el in feature_list[:int(n_total*train_ratio)]]
        valid_X = [el[0] for el in feature_list[int(n_total*train_ratio): int(n_total*(train_ratio+valid_ratio))]]
        valid_Y = [el[2] for el in feature_list[int(n_total*train_ratio): int(n_total*(train_ratio+valid_ratio))]]
        test_X = [el[0] for el in feature_list[int(n_total*(train_ratio+valid_ratio)):]]
        test_Y = [el[2] for el in feature_list[int(n_total*(train_ratio+valid_ratio)):]]

    elif data_unit == 'Pattern':
        random.shuffle(feature_list)
        random.shuffle(pattern_list)
        n_total = len(pattern_list)

        def select_pattern_sample(feature, patterns):
            selected_X = []
            selected_Y = []
            for el in feature:
                if el[4] in patterns:
                    selected_X.append(el[0])
                    selected_Y.append(el[2])
            return selected_X, selected_Y

        train_X , train_Y = select_pattern_sample(feature_list, pattern_list[:int(n_total*train_ratio)])
        valid_X, valid_Y = select_pattern_sample(feature_list, pattern_list[int(n_total*train_ratio): int(n_total*(train_ratio+valid_ratio))])
        test_X, test_Y = select_pattern_sample(feature_list, pattern_list[int(n_total*(train_ratio+valid_ratio)):])

    elif data_unit == 'ID':
        random.shuffle(feature_list)
        random.shuffle(id_list)
        n_total = len(id_list)

        def select_id_sample(feature, patterns):
            selected_X = []
            selected_Y = []
            for el in feature:
                if el[1] in patterns:
                    selected_X.append(el[0])
                    selected_Y.append(el[2])
            return selected_X, selected_Y

        train_X, train_Y = select_id_sample(feature_list, id_list[:int(n_total * train_ratio)])
        valid_X, valid_Y = select_id_sample(feature_list, id_list[int(n_total * train_ratio): int(
            n_total * (train_ratio + valid_ratio))])
        test_X, test_Y = select_id_sample(feature_list,
                                               id_list[int(n_total * (train_ratio + valid_ratio)):])
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test')
    parser.add_argument('--unit', default='id')
    args = parser.parse_args()

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = make_dataset()

    print ("{}, Data Unit: {}".format(args.name, args.unit))
    print (train_X.shape)
    print (test_X.shape)
    train_X = train_X.T
    test_X = test_X.T

    scaler = StandardScaler()
    scaler.fit(train_X)

    scaler_filename = "stat/scaler.save"
    joblib.dump(scaler, scaler_filename)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)


    mlp = MLPClassifier(hidden_layer_sizes=(10, 20, 10, 30))
    mlp.fit(train_X, train_Y)

    from sklearn.metrics import classification_report, confusion_matrix

    predictions = mlp.predict(test_X)

    print(confusion_matrix(test_Y, predictions))
    print(classification_report(test_Y, predictions))


    #test_MLP = [[20,10],[30,20,20,30,30],[10, 20, 10, 30],[1000,1000,1000],[30,30,30],[30, 30, 30, 30, 30, 30]]
    test_MLP = [[256,128,64]]




    for MLPT in test_MLP:
        mlp = MLPClassifier(hidden_layer_sizes=(MLPT))
        mlp.fit(train_X, train_Y)

        predictions = mlp.predict(test_X)

        print(confusion_matrix(test_Y, predictions))
        print(classification_report(test_Y, predictions))
        print (MLPT)
        pp=classification_report(test_Y, predictions)

    filename = '{}_model.sav'.format(args.name)
    pickle.dump(mlp, open(filename, 'wb'))

    # loaded_model = pickle.load(open('./finalized_model.sav', 'rb'))
    # result = loaded_model.score(test_X, test_Y)
    # print(result)
