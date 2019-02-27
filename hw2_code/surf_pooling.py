#!/bin/python
import numpy
import os
import cPickle
import sys
# Do average pooling on the surf kmeans feature

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} output_path, file_list".format(sys.argv[0])
        print "output_path -- path to the pooled kmeans features"
        print "file_list -- the list of videos"
        exit(1)

    output_path = sys.argv[1]
    file_list = sys.argv[2]

    fread = open(file_list, "r")
    for i, line in enumerate(fread.readlines()):
        kmeans_path = "kmeans/" + line.replace('\n', '') + ".npy"
        print(str(i) + " " + kmeans_path)
        # output file
        fwrite_path = output_path + line.replace('\n', '')

        # if there is no surf for the video
        if not os.path.exists(kmeans_path):
            continue
        else:
            kmeans_feat = numpy.load(kmeans_path)
            avg_pooled_feat = numpy.mean(kmeans_feat, axis=0)
            normed_feat = avg_pooled_feat / numpy.linalg.norm(avg_pooled_feat)
            numpy.save(fwrite_path, normed_feat)

    print "Feature pooling completed successfully!"
