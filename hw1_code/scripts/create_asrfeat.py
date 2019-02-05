#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)

    vocab_file = sys.argv[1]; file_list = sys.argv[2]

    # store all the vocab as an array
    fread = open(vocab_file, "r")
    lines = fread.readlines()
    vocab = []
    for line in lines:
        vocab.append(line.replace('\n', ''))
    fread.close()

    fread = open(file_list, "r")
    for line in fread.readlines():
        # for now, we just consider the text features
        # TODO: explore CTM file later
        txt_path = "asr/" + line.replace('\n', '') + ".txt"
        # output file
        fwrite = open("asrfeat/" + line.replace('\n', ''), "w")
        # initialize the histogram with all zeros
        hist = numpy.zeros(len(vocab))
        if os.path.exists(txt_path) == False:
            for i in xrange(len(vocab)):
                hist[i] = 1.0 / len(vocab)
        else:
            text_file = open(txt_path, "r")
            text = text_file.read()
            # remove punctuations and split on whitespaces and newlines
            words = text.replace(".", " ").replace(",", " ").split()
            total = 0
            for word in words:
                # convert to lowercase
                word = word.lower()
                if word in vocab:
                    hist[vocab.index(word)] += 1
                    total += 1
            text_file.close()
            # TODO: still, not sure if I need to normalize it
            if total != 0:
                for i in xrange(len(vocab)):
                    hist[i] /= len(total)

        # write the histogram into the file
        line = ";".join([str(x) for x in hist])
        fwrite.write(line + "\n")
        fwrite.close()

    print "ASR features generated successfully!"
