#!/bin/python
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        exit(1)

    folder = sys.argv[1] # mfcc_pred/ or asr_pred/
    feature_name = sys.argv[2]
    output_name = sys.argv[3]
    test_videos = open("../all_test.video", "r")

    file0 = open(folder + "NULL_" + feature_name + ".lst", "r")
    file1 = open(folder + "P001_" + feature_name + ".lst", "r")
    file2 = open(folder + "P002_" + feature_name + ".lst", "r")
    file3 = open(folder + "P003_" + feature_name + ".lst", "r")

    score0 = []
    score1 = []
    score2 = []
    score3 = []
    videos = []

    for line in file0.readlines():
        score0.append(float(line.replace('\n', '')))
    for line in file1.readlines():
        score1.append(float(line.replace('\n', '')))
    for line in file2.readlines():
        score2.append(float(line.replace('\n', '')))
    for line in file3.readlines():
        score3.append(float(line.replace('\n', '')))
    for line in test_videos.readlines():
        videos.append(line.replace('\n', ''))

    file0.close()
    file1.close()
    file2.close()
    file3.close()

    output_file = open(output_name, "wb")
    output_file.write("VideoID,Label")

    for i in range(len(score1)):
        line = "\n" + str(videos[i])
        if score1[i] > score2[i] and score1[i] > score3[i] and score1[i] > score0[i]:
            line += ",1"
        elif score2[i] > score1[i] and score2[i] > score3[i] and score2[i] > score0[i]:
            line += ",2"
        elif score3[i] > score1[i] and score3[i] > score2[i] and score3[i] > score0[i]:
            line += ",3"
        else:
            line += ",0"
        output_file.write(line)
    output_file.close()

    print "prediction file generated successfully!\n"


