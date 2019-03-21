## How to run the MED pipeline

#### How to run the classifier
``source run.pipeline.sh -m true -s false``

#### How to early fuse two features
``python2 early_fusion.py FEAT1_DIR/ FEAT2_DIR/ list/all.video OUTPUT_DIR/
``
For example, if I want to early fuse the MFCC feature and the Places365 feature, I can run the following command:
``python2 early_fusion.py mfcc_clusters/ places/ list/all.video mfcc_places/
``
After the early fusion, you can run the classifier as normal by specifying corresponding directory and parameters in the shell file.

#### How to late fuse two features
``python2 late_fusion.py FEAT1_DIR/ FEAT2_DIR/ IS_TEST(0|1) FEAT1_NAME FEAT2_NAME WEIGHT1 WEIGHT2
``
For example, if I want to late fuse the ResNet50 feature and the Places365 feature and assign them the same weight, I can run the following command:
``python2 late_fusion.py resnet_pred/ places_pred/ 0 resnet places 0.5 0.5
``
There is no need to run the pipeline shell file after late fusion since the late fusion is directly modifying the score files.
But you do need to run `generate_class.py` if you want to generate the file which can be submitted to Kaggle.

#### How to double fuse two features
``python2 double_fusion.py FEAT1_DIR/ FEAT2_DIR/ EF_DIR/ IS_TEST(0|1) FEAT1_NAME FEAT2_NAME EF_NAME WEIGHT1 WEIGHT2 EF_WEIGHT
``
For example, if I want to double fuse the ResNet50 feature and the MFCC feature, I can run the following command:
``python2 double_fusion.py resnet_pred/ mfcc_pred/ mfcc_resnet50_pred/ 0 resnet mfcc mfcc_resnet50 1 1 1
``