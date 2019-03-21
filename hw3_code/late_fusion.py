#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 8:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeansfeat_path -- the path of kmeans feature"
        print "asrfeat_path -- the path of asr feature"
        print "file_list -- the list of videos"
        print "output_path -- the name of the combined feature"
        exit(1)

    score1_path = sys.argv[1]  # kmeans/
    score2_path = sys.argv[2]  # asrfeat/
    is_test = bool(int(sys.argv[3]))
    feat1_name = sys.argv[4]
    feat2_name = sys.argv[5]
    weight1 = float(sys.argv[6])
    weight2 = float(sys.argv[7])

    if not is_test:
        ext1 = "_" + feat1_name + "_val.lst"
        ext2 = "_" + feat2_name + "_val.lst"
    else:
        ext1 = "_" + feat1_name + ".lst"
        ext2 = "_" + feat2_name + ".lst"

    NULL_file1 = open(score1_path + "NULL" + ext1, "r")
    P001_file1 = open(score1_path + "P001" + ext1, "r")
    P002_file1 = open(score1_path + "P002" + ext1, "r")
    P003_file1 = open(score1_path + "P003" + ext1, "r")
    files1 = [NULL_file1, P001_file1, P002_file1, P003_file1]

    NULL_file2 = open(score2_path + "NULL" + ext2, "r")
    P001_file2 = open(score2_path + "P001" + ext2, "r")
    P002_file2 = open(score2_path + "P002" + ext2, "r")
    P003_file2 = open(score2_path + "P003" + ext2, "r")
    files2 = [NULL_file2, P001_file2, P002_file2, P003_file2]

    output_path = feat1_name + "_" + feat2_name + "_LF_pred/"
    if not is_test:
        ext_out = "_LF_val.lst"
    else:
        ext_out = "_LF.lst"
    NULL_file_out = open(output_path + "NULL" + ext_out, "wb")
    P001_file_out = open(output_path + "P001" + ext_out, "wb")
    P002_file_out = open(output_path + "P002" + ext_out, "wb")
    P003_file_out = open(output_path + "P003" + ext_out, "wb")
    files_out = [NULL_file_out, P001_file_out, P002_file_out, P003_file_out]

    for i in range(4):
        score1 = files1[i].read().splitlines()
        score2 = files2[i].read().splitlines()

        for j, score in enumerate(score1):
            s1 = float(score1[j])
            s2 = float(score2[j])
            s = weight1 * s1 + weight2 * s2
            files_out[i].write(str(s) + "\n")

    print "Features late fused successfully!"