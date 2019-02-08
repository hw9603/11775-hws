#!/bin/bash

mkdir -p combined/

python scripts/create_asrfeat_kmeans.py kmeans/ asrfeat/ list/all.video;
