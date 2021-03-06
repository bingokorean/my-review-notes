from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# First convolutional layer
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 5, activation='relu')
network = max_pool_2d(network, 2)

# Second convolutional layer
network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 2)

# Fully-connected layer
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)

# Output layer
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# Train and Evaluate the Model
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
