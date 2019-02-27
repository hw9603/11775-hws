#!/bin/python

import numpy as np
import os
from sklearn.cluster.k_means_ import MiniBatchKMeans, KMeans
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} feat_path cluster_num output_file".format(sys.argv[0])
        print "feat_path -- path to all feature files"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    feat_path = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])

    fwrite = open(output_file, "wb")

    array = np.load(feat_path)

    print "define kmeans"
    kmeans = MiniBatchKMeans(n_clusters=cluster_num)

    print "fit kmeans"
    kmeans.fit(array)

    print "saving model"
    # save the model to the specified output file
    cPickle.dump(kmeans, fwrite, -1)

    print "K-means trained successfully!"
