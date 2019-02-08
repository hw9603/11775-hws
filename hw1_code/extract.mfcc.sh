#!/bin/bash

# An example script for feature extraction of Homework 1

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups.

# Paths to different tools; 
opensmile_path=/home/ubuntu/tools/opensmile-2.3.0/inst
export PATH=$opensmile_path/bin:$PATH
export LD_LIBRARY_PATH=$opensmile_path/lib:$LD_LIBRARY_PATH

# Two additional variables
video_path=~/video   # path to the directory containing all the videos. In this example setup, we are linking all the videos to "../video"

mkdir -p audio mfcc kmeans asrfeat

# This part does feature extraction, it may take quite a while if you have a lot of videos. Totally 3 steps are taken:
# 1. ffmpeg extracts the audio track from each video file into a wav file
# 2. The wav file may contain 2 channels. We always extract the 1st channel using ch_wave
# 3. SMILExtract generates the MFCC features for each wav file
#    The config file MFCC12_0_D_A.conf generates 13-dim MFCCs at each frame, together with the 1st and 2nd deltas. So you 
#    will see each frame totally has 39 dims. 
#    Refer to Section 2.5 of this document http://web.stanford.edu/class/cs224s/hw/openSMILE_manual.pdf for better configuration
#    (e.g., normalization) and other feature types (e.g., PLPs )     
cat list/train | awk '{print $1}' > list/train.video
cat list/val | awk '{print $1}' > list/val.video
cat list/train.video list/val.video list/test.video > list/all.video
for line in $(cat "list/all.video"); do
    ffmpeg -y -i $video_path/${line}.mp4 -ac 1 -f wav audio/$line.wav
    SMILExtract -C config/MFCC12_0_D_A.conf -I audio/$line.wav -O mfcc/$line.mfcc.csv
done
# You may find the number of MFCC files mfcc/*.mfcc.csv is slightly less than the number of the videos. This is because some of the videos
# don't have the audio track. For example, HVC1221, HVC1222, HVC1261, HVC1794

# In this part, we train a clustering model to cluster the MFCC vectors. In order to speed up the clustering process, we
# select a small portion of the MFCC vectors. In the following example, we only select 20% randomly from each video.
echo "Pooling MFCCs (optional)"
python scripts/select_frames.py list/train.video 0.2 select.mfcc.csv || exit 1;

# Great! We are done!
echo "SUCCESSFUL COMPLETION"
