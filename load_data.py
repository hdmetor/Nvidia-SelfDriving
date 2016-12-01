from __future__ import division

import os
import cv2
import numpy as np
import random


DATA_FOLDER = 'driving_dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

X = []
y = []

with open(TRAIN_FILE) as fp:
    for line in fp:
        path, angle = line.strip().split()
        full_path = os.path.join(DATA_FOLDER, path)
        X.append(full_path)
        y.append(float(angle))


total_images = len(X)

# better shuffle the images
# pairs = [(path, angle), ...]
pairs = list(zip(X, y))
random.shuffle(pairs)
X, y = zip(*pairs)

def return_data(split=.8):

    images = [np.float32(cv2.resize(cv2.imread(im, cv2.IMREAD_COLOR), (200, 66))) / 255 for im in X]
    split_index = int(split * len(X))
    train_X = images[:split_index]
    train_y = y[:split_index]

    test_X = images[split_index:]
    test_y = y[split_index:]

    return np.array(train_X), np.array(train_y), np.array(test_X), array.array(test_y)
