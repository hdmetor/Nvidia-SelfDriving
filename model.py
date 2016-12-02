from keras.layers import Convolution2D, Input
from keras.layers.core import Dense, Flatten, Lambda, Activation
from keras.models import Model, Sequential

import tensorflow as tf

from keras import backend as K

def atan_layer(x):
    print(x, tf.mul(tf.atan(x), 2))
    return tf.mul(tf.atan(x), 2)

def atan_layer_shape(input_shape):
    return input_shape

def atan(x):
    return tf.atan(x)

def NVIDA():

    inputs = Input(shape=(66, 200, 3))
    conv_1 = Convolution2D(24, 5, 5, activation='relu', name='conv_1', subsample=(2, 2))(inputs)
    conv_2 = Convolution2D(36, 5, 5, activation='relu', name='conv_2', subsample=(2, 2))(conv_1)
    conv_3 = Convolution2D(48, 5, 5, activation='relu', name='conv_3', subsample=(2, 2))(conv_2)

    conv_4 = Convolution2D(64, 3, 3, activation='relu', name='conv_4', subsample=(1, 1))(conv_3)
    conv_5 = Convolution2D(64, 3, 3, activation='relu', name='conv_5', subsample=(1, 1))(conv_4)

    flat = Flatten()(conv_5)

    dense_1 = Dense(1164)(flat)
    dense_2 = Dense(100, activation='relu')(dense_1)
    dense_3 = Dense(50, activation='relu')(dense_2)
    dense_4 = Dense(10, activation='relu')(dense_3)

    final = Dense(1, activation=atan)(dense_4)
    #angle = Lambda(lambda x: tf.mul(tf.atan(x), 2))(final)


    model = Model(input=inputs, output=final)
    model.compile(optimizer='Adam', loss='mse')

    return model


