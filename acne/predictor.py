#coding:utf-8

import cv2
import h5py
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility

batch_size = 32

class Predictor():
    def __init__(self, model_file='', weights_file=''):
        json_file = open(model_file, 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights(weights_file)

    def get_data(self, directory):
        data = np.empty((1, 1, 53, 53))
        img = cv2.imread(directory, 0)
        resize = cv2.resize(img, (53, 53))
        tmp = np.array(resize)
        data[0, 0, :, :] = tmp
        data = data.astype('float32')
        data /= 255
        return data

    def predict(self, dictory):
        print dictory
        data = self.get_data(dictory)
        classes = self.model.predict_classes(data, batch_size = batch_size) 
        return classes[0]


if __name__ == '__main__':
    predictor = Predictor(model_file = './keras_model/model.json', weights_file = './keras_model/weights.h5')
    print predictor.predict("/home/kingston/Downloads/cuochuang/baitou/se.jpg")
