#!/bin/python

if __name__ == '__main__':
    folder = "mfcc_pred/"
    test_videos = open("../all_test.video", "r")

    file1 = open(folder + "P001_mfcc.lst", "r")
    file2 = open(folder + "P002_mfcc.lst", "r")
    file3 = open(folder + "P003_mfcc.lst", "r")

    score1 = []
    score2 = []
    score3 = []
    videos = []

    for line in file1.readlines():
        score1.append(float(line.replace('\n', '')))
    for line in file2.readlines():
        score2.append(float(line.replace('\n', '')))
    for line in file3.readlines():
        score3.append(float(line.replace('\n', '')))
    for line in test_videos.readlines():
        videos.append(line.replace('\n', ''))

    file1.close()
    file2.close()
    file3.close()

    output_file = open("prediction.csv", "wb")
    output_file.write("VideoID,Label")

    for i in range(len(score1)):
        line = "\n" + str(videos[i])
        if score1[i] > score2[i] and score1[i] > score3[i]:
            line += ",1"
        elif score2[i] > score1[i] and score2[i] > score3[i]:
            line += ",2"
        else:
            line += ",3"
        output_file.write(line)
    output_file.close()

    print "prediction.csv generated successfully!\n"


