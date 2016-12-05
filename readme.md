# Nvidia SelfDriving

Keras implementation of [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) by Nvidia.

Many thanks to [Sully Chen](https://github.com/SullyChen) for the [original](https://github.com/SullyChen/Autopilot-TensorFlow) TensorFlow implementation and for sharing his own [dataset](https://drive.google.com/file/d/0B-KJCaaF7ellQUkzdkpsQkloenM/view?usp=sharing), used here for training.

# Network

The network consists of 5 convolution layers (three 5x5 and two 3x3), and 4 fully connected layers on top.

# Howto

To start training run `python main.py`. The data needs to be in the `driving_dataset` folder.
