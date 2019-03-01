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
    for i, line in enumerate(fread.readlines()):
        cnn_path = "cnn/" + line.replace('\n', '') + ".cnn.npy"
        print(str(i) + " " + cnn_path)
        # output file
        fwrite_path = "cnn_kmeans/" + line.replace('\n', '')

        if os.path.exists(fwrite_path + ".npy") == True:
            continue

        # if there is no surf for the video
        if os.path.exists(cnn_path) == False:
            hist = numpy.zeros(cluster_num)
            for i in xrange(cluster_num):
                hist[i] = 1.0 / cluster_num
        else:
            # initialize the histogram with all zeros
            hist = numpy.zeros(cluster_num)
            cnn_feat = numpy.load(cnn_path)
            if numpy.any(numpy.equal(cnn_feat, None)):
                for i in xrange(cluster_num):
                    hist[i] = 1.0 / cluster_num
            else:
                preds = kmeans.predict(cnn_feat)
                for p in preds:
                    hist[p] += 1.0 / len(preds)
        # write the histogram into the file
        numpy.save(fwrite_path, hist)

    print "K-means features generated successfully!"
