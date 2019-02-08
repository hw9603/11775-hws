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
        print "kmeansfeat_path -- the path of kmeans feature"
        print "asrfeat_path -- the path of asr feature"
        print "file_list -- the list of videos"
        exit(1)

    kmeansfeat_path = sys.argv[1] # kmeans/
    asrfeat_path = sys.argv[2] # asrfeat/
    file_list = sys.argv[3] # list/all.video

    fread = open(file_list, "r")

    for line in fread.readlines():
        fwrite = open("combined/" + line.replace('\n', ''), "wb")
        kmeans_file = open(kmeansfeat_path + line.replace('\n', ''), "r")
        asrfeat_file = open(asrfeat_path + line.replace('\n', ''), "r")
        kmeans = kmeans_file.readline().replace('\n', '')
        asrfeat = asrfeat_file.readline().replace('\n', '')
        kmeans_file.close()
        asrfeat_file.close()
        combined_feat = kmeans + ";" + asrfeat
        fwrite.write(combined_feat + "\n")
        fwrite.close()

    print "Features combined successfully!"
