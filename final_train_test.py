#-*- coding: utf-8 -*-
import sys
import os
import numpy as np
import librosa
from final_feature_summary import *
import librosa.display
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import pickle
import random
from sklearn.externals import joblib

FEATURE_PATH = './dataset/features/full'
SHUFFLE = True
DATA_UNIT = "ID"  # Unit of the data: subject, note are available.
NORM_UNIT = "pattern"

random.seed(634)

def mean_feature(feature):
    feature = np.reshape(np.mean(feature, axis=1),(-1,1))
    return feature
    
def normalize_features(features):
    means = []
    stds = []
    for feature in features:
        temp_mean = feature.mean(1)
        temp_std = feature.std(1)
        means.append(temp_mean)
        stds.append(temp_std)
    mean = np.mean(means, axis=0)
    std = np.std(stds, axis=0)
    for feature in features:
        feature[-1,:]=(feature[-1,:]-mean[-1])/(std[-1]+np.finfo(np.float32).eps)

def gather_features(feature_list, option="pattern"): #option: ID, pattern
    fidx_dict = {} 
    for fidx in range(len(feature_list)):
        feature = feature_list[fidx]
        if option == "pattern":
            if feature[1] in fidx_dict:
                if feature[4] in fidx_dict[feature[1]]:
                    fidx_dict[feature[1]][feature[4]].append(fidx)
                else:
                    fidx_dict[feature[1]][feature[4]] = [fidx]
            else:
                fidx_dict[feature[1]] = {feature[4]:[fidx]}
        elif option == "ID":
            if feature[1] in fidx_dict:
                fidx_dict[feature[1]].append(fidx)
            else:
                fidx_dict[feature[1]] = [fidx]
    return fidx_dict

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
    file_list = os.listdir(FEATURE_PATH)
    feature_list = []
    pattern_list = []
    id_list = []
    for el in file_list:
        id, label, beat_id, pattern_id = parse_name(el)
        feature_data = np.load(os.path.join(FEATURE_PATH, el))
        feature_list.append((feature_data, id, label, beat_id, pattern_id))
        pattern_list.append('{}_p{}'.format(id, pattern_id))
        id_list.append(id)

    pattern_list = list(set(pattern_list))
    id_list = list(set(id_list))

    return feature_list, pattern_list, id_list


def make_dataset(data_unit=DATA_UNIT, train_ratio=0.6, valid_ratio=0.2, norm_unit=NORM_UNIT):
    feature_list, pattern_list, id_list = load_data_feature()
    for i, feature in enumerate(feature_list):
        feature_list[i] = (mean_feature(feature[0]),)+feature[1:]
    fidx_dict = gather_features(feature_list, option=norm_unit)
    for key_id in fidx_dict.keys():
        id_dict = fidx_dict[key_id]
        if norm_unit == "pattern":
            for key_p in id_dict.keys():
                fidxs = id_dict[key_p]
                pf_list = []
                for lidx, fidx in enumerate(fidxs):
                    pf_list.append(feature_list[fidx][0])
                normalize_features(pf_list)

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
        test_X, text_Y = select_pattern_sample(feature_list, pattern_list[int(n_total*(train_ratio+valid_ratio)):, 0])

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
    train_X = np.asarray(train_X)
    train_X = np.squeeze(train_X,axis=2)
    train_Y = np.asarray(train_Y)
    valid_X = np.asarray(valid_X)
    valid_X = np.squeeze(valid_X,axis=2)
    valid_Y = np.asarray(valid_Y)
    test_X = np.asarray(test_X)
    test_X = np.squeeze(test_X,axis=2)
    test_Y = np.asarray(test_Y)
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

if __name__ == '__main__':

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = make_dataset()

    #####################
    from sklearn.preprocessing import StandardScaler


    print ("Shuffle: "+str(SHUFFLE)+"\t\t Data Unit: "+DATA_UNIT)
    print (train_X.shape)
    print (test_X.shape)
    #train_X = train_X.T
    #test_X = test_X.T


    scaler = StandardScaler()
    scaler.fit(train_X)

    scaler_filename = "stat/scaler.save"
    joblib.dump(scaler, scaler_filename)


    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    from sklearn.neural_network import MLPClassifier

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

    filename = 'finalized_model.sav'
    pickle.dump(mlp, open(filename, 'wb'))

    # loaded_model = pickle.load(open('./finalized_model.sav', 'rb'))
    # result = loaded_model.score(test_X, test_Y)
    # print(result)




#############################








    ##########SGD

    ###############feature normalizaiton original




    # train_X = train_X.T
    # train_X_mean = np.mean(train_X, axis=0)
    # train_X = train_X - train_X_mean
    # train_X_std = np.std(train_X, axis=0)
    # train_X = train_X / (train_X_std + 1e-5)
    #
    # valid_X = valid_X.T
    # valid_X = valid_X - train_X_mean
    # valid_X = valid_X/(train_X_std + 1e-5)
    #
    # # training model
    # alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    #
    # model = []
    # valid_acc = []
    # for a in alphas:
    #     clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a)
    #     model.append(clf)
    #     valid_acc.append(acc)
    #
    # # choose the model that achieve the best validation accuracy
    # final_model = model[np.argmax(valid_acc)]
    #
    # # now, evaluate the model with the test set
    # test_X = test_X.T
    # test_X = test_X - train_X_mean
    # test_X = test_X/(train_X_std + 1e-5)
    # test_Y_hat = final_model.predict(test_X)
    #
    # accuracy = np.sum((test_Y_hat == test_Y))/test_num_audio*100.0
    # print ('test accuracy = ' + str(accuracy) + ' %')
    #
    # filename = 'finalized_model.sav'
    # pickle.dump(final_model, open(filename, 'wb'))
    #
    #



##################


