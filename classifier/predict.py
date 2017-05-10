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

def load_model():
    json_file = open('./keras_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('./keras_model/weights.h5')
    return loaded_model


def get_data(directory):
    data = np.empty((1, 1, 53, 53))
    img = cv2.imread(directory, 0)
    resize = cv2.resize(img, (53, 53))
    tmp = np.array(resize)
    data[0, 0, :, :] = tmp
    data = data.astype('float32')
    data /= 255
    return data


def predict(dictory):
    model = load_model()
    data = get_data(dictory)
    classes = model.predict_classes(data, batch_size = batch_size) 
    return classes[0]


if __name__ == '__main__':
    print predict('/home/kingston/Downloads/cuochuang/baitou/sekuozeng.jpg')
