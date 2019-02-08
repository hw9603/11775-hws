#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import chi2_kernel
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        print ""
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    predict_test = bool(int(sys.argv[5]))

    clf = cPickle.load(open(model_file, "rb"))
    # validation dataset
    if predict_test == True:
        print "predict on test.video"
        file_list = "list/test.video"
    else:
        print "predict on val.video"
        file_list = "list/val.video"
    fread = open(file_list, "r")
    fwrite = open(output_file, "wb")

    """ for chi-squared kernel """
    train_file_list = "list/train"
    val_file_list = "list/val"
    train_videos = []

    train_file = open(train_file_list, "r")
    for line in train_file.readlines():
        file_name, event = line.replace('\n', '').split()
        train_videos.append(file_name)
    train_file.close()

    if predict_test == True:
        print "Also including validation file in training.\n"
        train_file = open(val_file_list, "r")
        for line in train_file.readlines():
            file_name, event = line.replace('\n', '').split()
            train_videos.append(file_name)
        train_file.close()

    train_feat = numpy.zeros([len(train_videos), feat_dim])
    for i, video in enumerate(train_videos):
        feature = numpy.genfromtxt(feat_dir + video, delimiter=";")
        # the feature shape should be the same as feat_dim
        if feature.shape[0] == feat_dim:
            train_feat[i, :] = feature
    """ end """

    videos = []
    for line in fread.readlines():
        file_name = line.replace('\n', '')
        feature = numpy.genfromtxt(feat_dir + file_name, delimiter=";")
        scores = clf.decision_function(chi2_kernel(feature.reshape(1, -1), train_feat))
        fwrite.write(str(scores[0]) + "\n")
    fwrite.close()

    print 'SVM tested successfully!'
