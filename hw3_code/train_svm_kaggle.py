#!/bin/python

import numpy
import os
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import cPickle
import sys

# Performs K-means clustering and save the model to a local file

event_encode = {"NULL": "0", "P001": "1", "P002": "2", "P003": "3"}

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print(len(sys.argv))
        print("Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0]))
        print("feat_dir -- dir of feature files")
        print("feat_dim -- dim of features")
        print("output_file -- path to save the svm model")
        print("include_val -- whether to include validation file in training")
        print("feat_extension -- the extension of the feature files")
        exit(1)

    feat_dir = sys.argv[1]
    feat_dim = int(sys.argv[2])
    output_file = sys.argv[3]
    include_val = bool(int(sys.argv[4]))
    feat_extension = sys.argv[5]

    # hardcode the train file path
    train_file_list = "list/train"
    val_file_list = "list/val"

    fread = open(train_file_list, "r")
    fwrite = open(output_file, "wb")
    # list of video names
    videos = []
    # output matrix (binary)
    y = []
    for line in fread.readlines():
        file_name, event = line.replace('\n', '').split()
        videos.append(file_name)
        # 1 if equal to the event_name, 0 otherwise
        y.append(int(event_encode[event]))
    fread.close()

    # include validation files in training...
    if include_val == True:
        print("Also including validation file in training.\n")
        fread = open(val_file_list, "r")
        for line in fread.readlines():
            file_name, event = line.replace('\n', '').split()
            videos.append(file_name)
            y.append(int(event_encode[event]))
        fread.close()

    # generate the input matrix
    X = numpy.zeros([len(videos), feat_dim])
    for i, video in enumerate(videos):
        if not os.path.exists(feat_dir + video + feat_extension):
            continue
        feature = numpy.load(feat_dir + video + feat_extension)
        # the feature shape should be the same as feat_dim
        if feature.shape[0] == feat_dim:
            X[i, :] = feature

    clf = SVC(decision_function_shape='ovr', kernel='precomputed', gamma='scale', C=1, class_weight='balanced')
    clf.fit(chi2_kernel(X), y)
    # clf.fit(laplacian_kernel(X), y)

    # clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma='scale', C=1)
    # clf.fit(X, y)

    # dump the model to the output file
    cPickle.dump(clf, fwrite, -1)

    print('SVM trained successfully!')
