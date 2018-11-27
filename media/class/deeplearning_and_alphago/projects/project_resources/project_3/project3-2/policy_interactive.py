# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
#
# Revision history
# Originally written in Matlab by Sae-Young Chung in Apr. 2016
#   for EE405C Electronics Design Lab <Network of Smart Systems>, Spring 2016
# Python & TensorFlow porting by Wonseok Jeon, Narae Ryu, Hwehee Chung in Dec. 2016
#   for EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016

import tensorflow as tf
import numpy as np
from numpy.random import rand
from numpy.random import randint as randi
from numpy.random import choice as random_sample
from numpy.random import shuffle 
from numpy import arange as nrange
from game import game1, game2, game3, game4, data_augmentation
import os
import sys
#####################################################################
"""                    DEFINE HYPERPARAMETERS                     """
#####################################################################
# Choose game.
game = game1()
# Initial Learning Rate
alpha_p = 2e-5 # learning rate for policy network
# size of minibatch
size_minibatch = 1024 
# training epoch
max_epoch = 10

#####################################################################
"""                COMPUTATIONAL GRAPH CONSTRUCTION               """
#####################################################################
def cross_entropy_loss_with_logit(Y, Y_):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Y, Y_)
    return tf.reduce_mean(cross_entropy)

def l2_regularization(loss, variables):
    for i in range(len(variables)):
        loss += 0.0001 * tf.nn.l2_loss(variables[i])
    return loss

def network_optimizer(Y, Y_, alpha, scope):
    # Cross entropy loss
    loss = cross_entropy_loss_with_logit(Y, Y_)
    # Parameters in this scope
    variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = scope)
    # L2 regularization
    loss = l2_regularization(loss, variables)
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(alpha).minimize(loss,\
            var_list = variables)
    return loss, optimizer

### NETWORK ARCHITECTURE ###
def value_network(state, nx, ny):
    # Set variable initializer.
    init = tf.random_normal_initializer(stddev = 0.0001)

    # Create variable named "weights1" and "biases1".
    weights1 = tf.get_variable("weights1", [3, 3, 3, 30], initializer = init)
    biases1 = tf.get_variable("biases1", [30], initializer = init)

    # Create 1st layer
    conv1 = tf.nn.conv2d(state, weights1, strides = [1, 1, 1, 1], padding = 'SAME')
    out1 = tf.nn.relu(conv1 + biases1)

    # Create variable named "weights2" and "biases2".
    weights2 = tf.get_variable("weights2", [3, 3, 30, 50], initializer = init)
    biases2 = tf.get_variable("biases2", [50], initializer = init)

    # Create 2nd layer
    conv2 = tf.nn.conv2d(out1, weights2, strides = [1, 1, 1, 1], padding ='SAME')
    out2 = tf.nn.relu(conv2 + biases2)
   
    # Create variable named "weights3" and "biases3".
    weights3 = tf.get_variable("weights3", [3, 3, 50, 70], initializer = init)
    biases3 = tf.get_variable("biases3", [70], initializer = init)

    # Create 3rd layer
    conv3 = tf.nn.conv2d(out2, weights3, strides = [1, 1, 1, 1], padding ='SAME')
    out3 = tf.nn.relu(conv3 + biases3)

    # Create variable named "weights4" and "biases4".
    weights1fc = tf.get_variable("weights1fc", [nx*ny*70, 100], initializer = init)
    biases1fc = tf.get_variable("biases1fc", [100], initializer = init)
    
    # Create 1st fully connected layer
    fc1 = tf.reshape(out3, [-1, nx*ny*70])
    out1fc = tf.nn.relu(tf.matmul(fc1, weights1fc) + biases1fc)

    # Create variable named "weights2fc" and "biases2fc".
    weights2fc = tf.get_variable("weights2fc", [100, 3], initializer = init)
    biases2fc = tf.get_variable("biases2fc", [3], initializer = init)

    # Create 2nd fully connected layer
    return tf.matmul(out1fc, weights2fc) + biases2fc

def policy_network(state, nx, ny):
    # Set variable initializer.
    init = tf.random_normal_initializer(stddev = 0.1)

    # Create variable named "weights1" and "biases1".
    weights1 = tf.get_variable("weights1", [3, 3, 3, 30], initializer = init)
    biases1 = tf.get_variable("biases1", [30], initializer = init)

    # Create 1st layer
    conv1 = tf.nn.conv2d(state, weights1, strides = [1, 1, 1, 1], padding = 'SAME')
    out1 = tf.nn.relu(conv1 + biases1)

    # Create variable named "weights2" and "biases2".
    weights2 = tf.get_variable("weights2", [3, 3, 30, 50], initializer = init)
    biases2 = tf.get_variable("biases2", [50], initializer = init)

    # Create 2nd layer
    conv2 = tf.nn.conv2d(out1, weights2, strides = [1, 1, 1, 1], padding ='SAME')
    out2 = tf.nn.relu(conv2 + biases2)
   
    # Create variable named "weights3" and "biases3".
    weights3 = tf.get_variable("weights3", [3, 3, 50, 70], initializer = init)
    biases3 = tf.get_variable("biases3", [70], initializer = init)

    # Create 3rd layer
    conv3 = tf.nn.conv2d(out2, weights3, strides = [1, 1, 1, 1], padding ='SAME')
    out3 = tf.nn.relu(conv3 + biases3)

    # Create variable named "weights4" and "biases4"
    weights4 = tf.get_variable("weights4", [1, 1, 70, 1], initializer = init)
    biases4 = tf.get_variable("biases4", [nx, ny, 1], initializer = init)
    
    # Create 4th layer
    conv4 = tf.nn.conv2d(out3, weights4, strides = [1, 1, 1, 1], padding ='SAME')

    return tf.reshape(conv4 + biases4, [-1, nx*ny])


### NETWORK CONSTRUCTION ###
"""
    value_network0, policy_network0 for black
    value_network1, policy_network1 for white
"""
Loss = []; Optimizer = []

for i in range(2):

    # Define value network.
    scope = "value_network" + str(i)

    with tf.variable_scope(scope):
        # Input
        S = tf.placeholder(tf.float32, shape = [None, game.nx, game.ny, 3], name = "S")
        # Estimation for unnormalized log probability
        Y = value_network(S, game.nx, game.ny) 
        # Estimation for probability
        P = tf.nn.softmax(Y, name = "softmax")
        # Target in integer
        W = tf.placeholder(tf.int32, shape = [None], name = "W")
        # Target in one-hot vector
        Y_= tf.one_hot(W, 3, name = "Y_")

    # Define policy network.
    scope = "policy_network" + str(i) 

    with tf.variable_scope(scope):
        # Input
        S = tf.placeholder(tf.float32, shape = [None, game.nx, game.ny, 3], name = "S")
        # Estimation for unnormalized log probability
        Y = policy_network(S, game.nx, game.ny)
        # Estimation for probability
        P = tf.nn.softmax(Y, name = "softmax")
        # Target in one-hot vector
        Y_ = tf.placeholder(tf.float32, shape = [None, game.nx * game.ny], name = "Y_")
        # Define loss and optimizer for policy network (for both SL and RL)
#        loss, optimizer = network_optimizer(Y, Y_, alpha_p, scope)
        # Append loss and optimizer.
#        Loss.append(loss); Optimizer.append(optimizer)
    
V0_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "value_network0")
V1_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "value_network1")
P0_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "policy_network0")
P1_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "policy_network1")

### SAVER ###
saver0 = tf.train.Saver(V0_variables, max_to_keep = 10000)
saver1 = tf.train.Saver(V1_variables, max_to_keep = 10000)
saver2 = tf.train.Saver(P0_variables, max_to_keep = 10000)
saver3 = tf.train.Saver(P1_variables, max_to_keep = 10000)

with tf.Session() as sess:
    ### DEFAULT SESSION ###
    sess.as_default()

    ### VARIABLE INITIALIZATION ###
    sess.run(tf.initialize_all_variables())
  
    saver2.restore(sess, sys.argv[1])
    saver3.restore(sess, sys.argv[2])
    P1 = tf.get_default_graph().get_tensor_by_name("policy_network0/softmax:0")
    P2 = tf.get_default_graph().get_tensor_by_name("policy_network1/softmax:0")
    game.play_interactive([], [P1,P2], 0, [], [], 0) 
