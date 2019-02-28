#!/bin/python
import numpy
import os
import cPickle
import sys
# Do average pooling on the cnn feature

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} output_path, file_list".format(sys.argv[0])
        print "output_path -- path to the pooled cnn features"
        print "file_list -- the list of videos"
        exit(1)

    output_path = sys.argv[1]
    file_list = sys.argv[2]

    fread = open(file_list, "r")
    for i, line in enumerate(fread.readlines()):
        cnn_path = "cnn/" + line.replace('\n', '') + ".cnn.npy"
        print(str(i) + " " + cnn_path)
        # output file
        fwrite_path = output_path + line.replace('\n', '')

        # if there is no surf for the video
        if not os.path.exists(cnn_path):
            continue
        else:
            cnn_feat = numpy.load(cnn_path)
            if numpy.any(numpy.equal(cnn_feat, None)):
                print(cnn_path + " contains none. Assigning same value...")
                normed_feat = numpy.full(1000, 1/1000)
            else:
                avg_pooled_feat = numpy.mean(cnn_feat, axis=0)
                normed_feat = avg_pooled_feat / numpy.linalg.norm(avg_pooled_feat)
            numpy.save(fwrite_path, normed_feat)

    print "Feature pooling completed successfully!"
