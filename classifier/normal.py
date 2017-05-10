#coding:utf-8

import os
import sys
import numpy
import cv2
import cPickle

def getDirLabel():
    root = './cuochuang'
    trainDir = []
    trainLabel = []
    testDir = []
    testLabel = []
    idx = 0
    for i in os.listdir(root):
        dirName = root + '/' + i
        idx += 1
        files = os.listdir(dirName)
        cut = int(round(len(files) * 0.3))
        for j in range(len(files)):
            fileName = dirName + '/' + files[j]
            if j<cut:
                testDir.append(fileName)
                testLabel.append(idx)
            else:
                trainDir.append(fileName)
                trainLabel.append(idx)
    print len(trainDir)
    print len(testDir)
    return trainDir, trainLabel, testDir, testLabel

def getDataLabels(directory, label):
    num = len(directory)
    data = numpy.empty((num, 1, 53, 53))
    labels = numpy.empty(num)
    idx = 0
    for i in directory:
        img = cv2.imread(i, 0)
        print i
        resize = cv2.resize(img, (53, 53))
        tmp = numpy.array(resize)
        data[idx, 0, :, :] = tmp
        labels[idx] = label[idx]
        idx += 1
    return data, labels

def write2File(data, labels, name):
    writeFile = open(name, 'wb')
    cPickle.dump(data, writeFile, -1)
    cPickle.dump(labels, writeFile, -1)
    writeFile.close()

def main():
    import pdb; pdb.set_trace()
    
    trainDir, trainLabel, testDir, testLabel= getDirLabel()
    trainData, trainLabels = getDataLabels(trainDir, trainLabel)
    print "train data read"
    testData, testLabels = getDataLabels(testDir, testLabel)
    print "test data read"
    write2File(trainData, trainLabels, './train.pkl')
    print "train data write"
    write2File(testData, testLabels, './test.pkl')
    print "test data write"


main()
