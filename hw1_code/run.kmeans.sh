#!/bin/bash

cluster_num=1000        # the number of clusters in k-means. Note that 50 is by no means the optimal solution.
                      # You need to explore the best config by yourself.

# now trains a k-means model using the sklearn package
echo "Training the k-means model"
python scripts/train_kmeans.py select.mfcc.csv $cluster_num kmeans.${cluster_num}.model || exit 1;

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters.
# Each video is represented by a single vector which has the same dimension as the number of clusters.
echo "Creating k-means cluster vectors"
python scripts/create_kmeans.py kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;

# Now you can see that you get the bag-of-word representations under kmeans/. Each video is now represented
# by a {cluster_num}-dimensional vector.