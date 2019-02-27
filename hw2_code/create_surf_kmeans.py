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
        surf_path = "surf/" + line.replace('\n', '') + ".surf.npy"
        print(str(i) + " " + surf_path)
        # output file
        fwrite_path = "kmeans/" + line.replace('\n', '')

        if os.path.exists(fwrite_path + ".npy") == True:
            continue

        # if there is no surf for the video
        if os.path.exists(surf_path) == False:
            hist = numpy.zeros((1, cluster_num))
            for i in xrange(cluster_num):
                hist[0][i] = 1.0 / cluster_num
        else:
            surf_feat = numpy.load(surf_path)
            # initialize the histogram with all zeros
            hist = numpy.zeros((surf_feat.shape[0], cluster_num))
            for j, frame in enumerate(surf_feat):
                preds = kmeans.predict(frame)
                for p in preds:
                    hist[j][p] += 1.0 / len(preds)
        # write the histogram into the file
        numpy.save(fwrite_path, hist)

    print "K-means features generated successfully!"
