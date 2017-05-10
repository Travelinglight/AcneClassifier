#coding:utf-8

import cv2
import h5py
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility

batch_size = 32
nb_classes = 7

# input image dimensions
img_rows, img_cols = 53, 53
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def def_model():
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


def getData(directory):
    data = np.empty((1, 1, 53, 53))
    img = cv2.imread(directory, 0)
    resize = cv2.resize(img, (53, 53))
    tmp = np.array(resize)
    data[0, 0, :, :] = tmp
    data = data.astype('float32')
    data /= 255
    return data


def getLabel(data):
    model = def_model()
    model.load_weights('my_model_weights.h5')
    classes = model.predict_classes(data, batch_size = batch_size) 
    return classes


def predict(dictory):
    return getLabel(getData(dictory))[0]


if __name__ == '__main__':
    print predict('/home/kingston/Downloads/cuochuang/baitou/sekuozeng.jpg')
