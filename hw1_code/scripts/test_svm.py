#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    clf = cPickle.load(open(model_file, "rb"))
    # validation dataset
    file_list = "list/val.video"
    fread = open(file_list, "r")
    fwrite = open(output_file, "wb")
    videos = []
    for line in fread.readlines():
        file_name = line.replace('\n', '')
        feature = numpy.genfromtxt(feat_dir + file_name, delimiter=";")
        scores = clf.decision_function(feature.reshape(1, -1))
        fwrite.write(str(scores[0]) + "\n")
    fwrite.close()

    print 'SVM tested successfully!'
