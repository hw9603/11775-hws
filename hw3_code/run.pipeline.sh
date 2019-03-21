#!/bin/bash

while getopts s:m: option		# d:p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	s) SINGLE=${OPTARG};;                 # boolean
	m) MULTI=${OPTARG};;
	esac
	done

#export PATH=~/anaconda3/bin:$PATH

surf_cluster_num=400
cnn_cluster_num=1000

if [ "$SINGLE" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    feat_dim_cnn=1000
    feat_dim_resnet50=2048
    feat_dim_places=4096
    feat_dim_resnet50_places=6144
    feat_dim=${feat_dim_resnet50}
    mkdir -p cnn_pred/
    mkdir -p resnet_pred/
    mkdir -p places_pred/
    mkdir -p resnet_places_pred/
    pred_dir=resnet_pred
    feat_dir=resnet50/

    feat_name=resnet
    feat_extension=.npy

    # 1. TODO: Train SVM with OVR using only videos in training set.
    echo "Train SVM with OVR using only videos in training set."
    for event in P001 P002 P003 NULL; do
      echo "=========  Event $event  ========="
      python2 train_svm.py ${event} ${feat_dir} ${feat_dim} ${pred_dir}/svm.${event}.val.model 0 ${feat_extension};
    done

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
    echo "Test SVM with val set and calculate its MAP scores for own info."
    for event in P001 P002 P003 NULL; do
      echo "=========  Event $event  ========="
      python2 test_svm.py ${pred_dir}/svm.${event}.val.model ${feat_dir} ${feat_dim} ${pred_dir}/${event}_${feat_name}_val.lst 0 ${feat_extension} ${event};
      #  ap list/${event}_val_label surf_pred/${event}_surf_val.lst
      python2 evaluator.py list/${event}_val_label ${pred_dir}/${event}_${feat_name}_val.lst
    done

	# 3. TODO: Train SVM with OVR using videos in training and validation set.
	echo "Train SVM with OVR using videos in training and validation set."
    for event in P001 P002 P003 NULL; do
      echo "=========  Event $event  ========="
      python2 train_svm.py ${event} ${feat_dir} ${feat_dim} ${pred_dir}/svm.${event}.model 1 ${feat_extension};
    done

	# 4. TODO: Test SVM with test set saving scores for submission
	echo "Test SVM with test set saving scores for submission"
	for event in P001 P002 P003 NULL; do
      echo "=========  Event $event  ========="
      python2 test_svm.py ${pred_dir}/svm.${event}.model ${feat_dir} ${feat_dim} ${pred_dir}/${event}_${feat_name}.lst 1 ${feat_extension} ${event};
    done
    python2 generate_class.py ${pred_dir}/ ${feat_name} prediction_${feat_name}.csv
fi

if [ "$MULTI" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    feat_dim_cnn=1000
    feat_dim_resnet50=2048
    feat_dim_places=4096
    feat_dim_resnet50_places=6144
    feat_dim_mfcc=200
    feat_dim_mfcc_resnet50=2248
    feat_dim_mfcc_places=4296
    feat_dim=${feat_dim_mfcc_resnet50}
    mkdir -p cnn_pred/
    mkdir -p resnet_pred/
    mkdir -p places_pred/
    mkdir -p resnet_places_pred/
    mkdir -p mfcc_pred/
    mkdir -p mfcc_resnet50_pred/
    mkdir -p mfcc_places_pred/
    pred_dir=mfcc_resnet50_pred
    feat_dir=mfcc_resnet50/

    feat_name=mfcc_resnet50
    feat_extension=.npy

    # 1. TODO: Train SVM with OVR using only videos in training set.
    echo "Train SVM with OVR using only videos in training set."
    python2 train_svm_kaggle.py ${feat_dir} ${feat_dim} ${pred_dir}/svm.val.model 0 ${feat_extension};

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
    echo "Test SVM with val set and calculate its MAP scores for own info."
    python2 test_svm_kaggle.py ${pred_dir}/svm.val.model ${feat_dir} ${feat_dim} prediction_${feat_name}.csv 0 ${feat_extension} 1 ${feat_name} ${pred_dir}/;
    for event in P001 P002 P003 NULL; do
      echo "=========  Event $event  ========="
      #  ap list/${event}_val_label surf_pred/${event}_surf_val.lst
      python2 evaluator.py list/${event}_val_label ${pred_dir}/${event}_${feat_name}_val.lst
    done

	# 3. TODO: Train SVM with OVR using videos in training and validation set.
	echo "Train SVM with OVR using videos in training and validation set."
    python2 train_svm_kaggle.py ${feat_dir} ${feat_dim} ${pred_dir}/svm.model 1 ${feat_extension};

	# 4. TODO: Test SVM with test set saving scores for submission
	echo "Test SVM with test set saving scores for submission"
    python2 test_svm_kaggle.py ${pred_dir}/svm.model ${feat_dir} ${feat_dim} prediction_${feat_name}.csv 1 ${feat_extension} 1 ${feat_name} ${pred_dir}/;
fi