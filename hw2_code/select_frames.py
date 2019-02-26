#!/bin/python
# Randomly select

import numpy as np
import os
import sys
import random

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} file_list select_ratio output_file".format(sys.argv[0])
        print "file_list -- the list of feature files"
        print "select_ratio -- the ratio of frames to be randomly selected from each audio file"
        print "output_file -- path to save the selected frames (feature vectors)"
        exit(1)

    file_list = sys.argv[1]; output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list, "r")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first
    # num_of_frame * ratio rows
    np.random.seed(18877)

    selected_array = None

    for i, line in enumerate(fread.readlines()):
        surf_path = "surf/" + line.replace('\n', '') + ".surf.npy"
        print(str(i) + " " + surf_path)
        if os.path.exists(surf_path) == False:
            print(surf_path + " does not exist. Skipping...")
            continue
        array = np.load(surf_path)
        array_2d = None

        for j in range(array.shape[0]):
            index = random.sample(range(0, array[j].shape[0]), int(array[j].shape[0] * ratio))
            if array_2d is None:
                array_2d = array[j][index, ]
            else:
                array_2d = np.vstack((array_2d, array[j][index, ]))

        if selected_array is None:
            selected_array = array_2d
        else:
            selected_array = np.vstack((selected_array, array_2d))

    np.save(output_file, selected_array)

