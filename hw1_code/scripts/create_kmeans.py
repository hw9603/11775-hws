#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model, "rb"))

    fread = open(file_list, "r")
    for line in fread.readline():
        mfcc_path = "mfcc/" + line.replace('\n', '') + ".mfcc.csv"
        # output file
        fwrite = open("kmeans/" + line.replace('\n', ''), "w")
        # initialize the histogram with all zeros
        hist = numpy.zeros(cluster_num)
        # if there is no mfcc for the video
        if os.path.exists(mfcc_path) == False:
            for i in xrange(cluster_num):
                hist[i] = 1.0 / cluster_num
        else:
            array = numpy.genfromtxt(mfcc_path, delimiter=";")
            preds = kmeans.predict(array)
            for p in preds:
                # TODO: compare with unnormalized version
                hist[p] += 1.0 / len(preds)
        # write the histogram into the file
        line = ";".join(hist)
        fwrite.write(line + "\n")
        fwrite.close()

    print "K-means features generated successfully!"
