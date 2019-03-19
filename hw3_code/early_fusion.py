#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeansfeat_path -- the path of kmeans feature"
        print "asrfeat_path -- the path of asr feature"
        print "file_list -- the list of videos"
        print "output_path -- the name of the combined feature"
        exit(1)

    feat1_path = sys.argv[1]  # kmeans/
    feat2_path = sys.argv[2]  # asrfeat/
    file_list = sys.argv[3]  # list/all.video
    output_path = sys.argv[4]

    fread = open(file_list, "r")

    for line in fread.readlines():
        feat1 = np.load(feat1_path + line.replace('\n', '') + ".npy")
        feat2 = np.load(feat2_path + line.replace('\n', '') + ".npy")
        combined_feat = np.concatenate((feat1, feat2), axis=None)
        np.save(output_path + line.replace('\n', ''), combined_feat)

    print "Features combined successfully!"
