#!/bin/python

import numpy
import os
from sklearn.cluster.k_means_ import MiniBatchKMeans, KMeans
import cPickle
import sys
import pandas

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    mfcc_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])

    fwrite = open(output_file, "wb")

    print "begin generate array"
    selection = pandas.read_csv(mfcc_csv_file, sep=';', dtype='float')
    array = selection.values
    # array = numpy.genfromtxt(mfcc_csv_file, delimiter=";")

    print "define kmeans"
    kmeans = MiniBatchKMeans(n_clusters=cluster_num)

    print "fit kmeans"
    kmeans.fit(array)

    print "saving model"
    # save the model to the specified output file
    cPickle.dump(kmeans, fwrite, -1)

    print "K-means trained successfully!"
