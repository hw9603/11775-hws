#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        print "include_val -- whether to include validation file in training"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    include_val = bool(int(sys.argv[5]))

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
        if event == "NULL":
            continue
        videos.append(file_name)
        # 1 if equal to the event_name, 0 otherwise
        y.append(int(event == event_name))
    fread.close()

    # include validation files in training...
    if include_val == True:
        print "Also including validation file in training.\n"
        fread = open(val_file_list, "r")
        for line in fread.readlines():
            file_name, event = line.replace('\n', '').split()
            if event == "NULL":
                print "ignoring null..."
                continue
            videos.append(file_name)
            y.append(int(event == event_name))
        fread.close()

    # generate the input matrix
    X = numpy.zeros([len(videos), feat_dim])
    for i, video in enumerate(videos):
        if not os.path.exists(feat_dir + video + ".npy"):
            continue
        feature = numpy.load(feat_dir + video + ".npy")
        # the feature shape should be the same as feat_dim
        if feature.shape[0] == feat_dim:
            X[i, :] = feature

    clf = SVC(decision_function_shape='ovr', kernel='precomputed', gamma='scale', C=1)
    clf.fit(chi2_kernel(X), y)

    # clf = SVC(decision_function_shape='ovr', kernel='rbf', gamma='scale', C=1)
    # clf.fit(X, y)

    # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=100, learning_rate=0.1)
    # clf.fit(X, y)
    # dump the model to the output file
    cPickle.dump(clf, fwrite, -1)

    print 'SVM trained successfully for event %s!' % (event_name)
