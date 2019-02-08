#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups. 

# Paths to different tools; 
map_path=/home/ubuntu/tools/mAP
export PATH=$map_path:$PATH

echo "#####################################"
echo "#       MED with MFCC Features      #"
echo "#####################################"
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=1000
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "kmeans/" $feat_dim_mfcc mfcc_pred/svm.$event.model 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python scripts/test_svm.py mfcc_pred/svm.$event.model "kmeans/" $feat_dim_mfcc mfcc_pred/${event}_mfcc.lst 1;
done

echo ""
echo "####################################"
echo "#    Generate Prediction File      #"
echo "####################################"
python generate_class.py mfcc_pred/ mfcc

echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
mkdir -p asr_pred
# iterate over the events
feat_dim_asr=8546
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "asrfeat/" $feat_dim_asr asr_pred/svm.$event.model 1 || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python scripts/test_svm.py asr_pred/svm.$event.model "asrfeat/" $feat_dim_asr asr_pred/${event}_asr.lst 1 || exit 1;
done

echo ""
echo "####################################"
echo "#    Generate Prediction File      #"
echo "####################################"
python generate_class.py asr_pred/ asr

echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
mkdir -p combined_pred
# iterate over the events
feat_dim_combined=9546
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "combined/" $feat_dim_combined combined_pred/svm.$event.model 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python scripts/test_svm.py combined_pred/svm.$event.model "combined/" $feat_dim_combined combined_pred/${event}_combined.lst 1;
done

echo ""
echo "####################################"
echo "#    Generate Prediction File      #"
echo "####################################"
python generate_class.py combined_pred/ combined