#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from scipy.misc import imresize

def get_cnn_features_from_video(downsampled_video_filename, cnn_feat_video_filename, keyframe_interval):
    """
    Receives filename of downsampled video and of output path for features.
    Extracts features in the given keyframe_interval.
    Saves features in pickled file.
    """
    keyframes = get_keyframes(downsampled_video_filename, keyframe_interval)

    cnn_features = None
    for image in keyframes:
        image = imresize(image, (224, 224))
        x = np.array(image, dtype=np.float64)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        if cnn_features is None:
            cnn_features = features
        else:
            cnn_features = np.vstack((cnn_features, features))
    np.save(cnn_feat_video_filename, cnn_features)


def get_keyframes(downsampled_video_filename, keyframe_interval):
    """ Generator function which returns the next keyframe. """

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    cnn_features_folderpath = my_params.get('cnn_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: define CNN object
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    # Check if folder for CNN features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get CNN features for all videos but only from keyframes

    fread = open(all_video_names, "r")

    for i, line in enumerate(fread.readlines()):
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        cnn_feat_video_filename = os.path.join(cnn_features_folderpath, video_name + '.cnn')

        print(str(i) + " " + video_name)
        if not os.path.isfile(downsampled_video_filename):
            continue

        # Get CNN features for one video
        try:
            get_cnn_features_from_video(downsampled_video_filename,
                                        cnn_feat_video_filename, keyframe_interval)
        except:
            print("generate CNN feature error for file {0}".format(video_name))
