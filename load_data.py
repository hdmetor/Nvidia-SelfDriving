from __future__ import division

import os
import numpy as np
import random

from scipy import pi
from scipy.misc import imread, imresize

from itertools import islice

LIMIT = None

DATA_FOLDER = 'driving_dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

# the expected dimension is (tf ordering)
# (None, 66, 200, 3)
def return_data(split=.8):

    X = []
    y = []

    with open(TRAIN_FILE) as fp:
        for line in islice(fp, LIMIT):
            path, angle = line.strip().split()
            full_path = os.path.join(DATA_FOLDER, path)
            X.append(full_path)
            # using angles from -pi to pi to avoid rescaling the atan in the network
            y.append(float(angle) * pi / 180 / 2)
    y = np.array(y)


    images = np.array([np.float32(imresize(imread(im), size=(66, 200))) / 255 for im in X])
    split_index = int(split * len(X))

    train_X = images[:split_index]
    train_y = y[:split_index]
    test_X = images[split_index:]
    test_y = y[split_index:]

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
