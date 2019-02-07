#!/bin/bash

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand. Each video is represented by
# a vector which has the same dimension as the size of the vocabulary. The elements of this vector are the number of occurrences
# of the corresponding word. The vector is normalized to be like a probability.
# You can of course explore other better ways, such as TF-IDF, of generating these features.
echo "Creating ASR features"
# mkdir -p asrfeat
python scripts/create_asrfeat.py vocab list/all.video;
