#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores, d) computation of class labels for kaggle submission.

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -d true -p true -f true -m true -k true -y filepath

# Reading of all arguments:
while getopts d:p:f:m:k:y: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	d) DOWNSAMPLING=${OPTARG};;        # boolean true or false
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

export PATH=~/anaconda3/bin:$PATH

if [ "$DOWNSAMPLING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=/home/ubuntu/11775_videos/video  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../hw1_code/list/val > list/val.video
    cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`

#    find . -name "*jpeg" | parallel -I% --max-args 1 convert % %.png

    cat "list/all.video" | parallel --jobs 32 -I% --max-args 1 ffmpeg -y -ss 0 -i $video_path/%.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/%.ds.mp4

#    for line in $(cat "list/all.video"); do
#        ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
#    done
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization

fi

if [ "$PREPROCESSING" = true ] ; then

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    python surf_feat_extraction.py -i list/all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
    python cnn_feat_extraction.py -i list/all.video config.yaml
fi

surf_cluster_num = 400

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"
    # 0. TODO: Randomly select 10% features
    python select_frames.py list/train.video 0.1 select.surf

    # 1. TODO: Train kmeans to obtain clusters for SURF features
    python train_surf_kmeans.py select.surf.npy $surf_cluster_num surf.kmeans.${surf_cluster_num}.model

    # 2. TODO: Create kmeans representation for SURF features
    python create_surf_kmeans.py surf.kmeans.${surf_cluster_num}.model $surf_cluster_num list/all.video

    # 3. TODO: Average/Max-pooling over BOW SURF feature and do normalization
    mkdir -p pool_kmeans/
    python surf_pooling.py pool_kmeans/ list/all.video

    echo "#####################################"
    echo "#   CNN FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for CNN features


    # 2. TODO: Create kmeans representation for CNN features

fi

if [ "$MAP" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    feat_dim_surf=400
    mkdir -p surf_pred

    # 1. TODO: Train SVM with OVR using only videos in training set.
    echo "Train SVM with OVR using only videos in training set."
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python train_svm.py $event "pool_kmeans/" $feat_dim_surf surf_pred/svm.$event.val.model 0;
    done

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
    echo "Test SVM with val set and calculate its MAP scores for own info."
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python test_svm.py surf_pred/svm.$event.val.model "pool_kmeans/" $feat_dim_surf surf_pred/${event}_surf_val.lst 0;
      #  ap list/${event}_val_label surf_pred/${event}_surf_val.lst
      python evaluator.py list/${event}_val_label surf_pred/${event}_surf_val.lst
    done

	# 3. TODO: Train SVM with OVR using videos in training and validation set.
	echo "Train SVM with OVR using videos in training and validation set."
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python train_svm.py $event "pool_kmeans/" $feat_dim_surf surf_pred/svm.$event.model 1;
    done

	# 4. TODO: Test SVM with test set saving scores for submission
	echo "Test SVM with test set saving scores for submission"
	for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python test_svm.py surf_pred/svm.$event.model "pool_kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst 1;
    done

    echo "#######################################"
    echo "# MED with CNN Features: MAP results  #"
    echo "#######################################"


    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

fi


if [ "$KAGGLE" = true ] ; then

    echo "##########################################"
    echo "# MED with SURF Features: KAGGLE results #"
    echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.
    echo "Train SVM with OVR using only videos in training set."
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python train_svm.py $event "pool_kmeans/" $feat_dim_surf surf_pred/svm.$event.val.model 0;
    done

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
    echo "Test SVM with val set and calculate its MAP scores for own info."
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python test_svm.py surf_pred/svm.$event.val.model "pool_kmeans/" $feat_dim_surf surf_pred/${event}_surf_val.lst 0;
      #  ap list/${event}_val_label surf_pred/${event}_surf_val.lst
      python evaluator.py list/${event}_val_label surf_pred/${event}_surf_val.lst
    done

	# 3. TODO: Train SVM with OVR using videos in training and validation set.
	echo "Train SVM with OVR using videos in training and validation set."
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python train_svm.py $event "pool_kmeans/" $feat_dim_surf surf_pred/svm.$event.model 1;
    done

    # 4. TODO: Test SVM with test set saving scores for submission
    echo "Test SVM with test set saving scores for submission"
	for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      python test_svm.py surf_pred/svm.$event.model "pool_kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst 1;
    done
    python generate_class.py surf_pred/ surf

    echo "##########################################"
    echo "# MED with CNN Features: KAGGLE results  #"
    echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

fi
