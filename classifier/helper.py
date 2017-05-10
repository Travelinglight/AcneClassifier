#coding:utf-8

import os
import sys
import numpy
import cv2


def getData(directory):
    data = numpy.empty((1, 53, 53))
    img = cv2.imread(directory, 0)
    resize = cv2.resize(img, (53, 53))
    tmp = numpy.array(resize)
    data[0, :, :] = tmp
    return data


def main():
    print getData('./cuochuang/baitou/799279068857524627_\xe7\x9c\x8b\xe5\x9b\xbe\xe7\x8e\x8bkuozeng.jpg')

main()
