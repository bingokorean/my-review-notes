﻿# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
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
        loss, optimizer = network_optimizer(Y, Y_, alpha_p, scope)
        # Append loss and optimizer.
        Loss.append(loss); Optimizer.append(optimizer)
    
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

    win = np.zeros((2))
    loss = np.zeros((2))
    tie = np.zeros((2))

    ### VARIABLE INITIALIZATION ###
    sess.run(tf.initialize_all_variables())
    
    # number of games to play for training
    n_train = 10000
    # number of games to play for testing
    n_test = 100
    
    # Play 'ng' games between two players using the previous
    mt = np.floor(game.nx * game.ny / 2)
    r1r = rand(n_train, 1)
    r2r = rand(n_train, 1)
    r1k = randi(mt * 2, size = (n_train, 1))
    r2k = randi(mt * 2, size = (n_train, 1))
    r1 = (r1k > mt) * r1r + (r1k <= mt) * (-r1k)
    r2 = (r2k > mt) * r2r + (r2k <= mt) * (-r2k)
   
    # Load the pre-trained 'value network'
    saver0.restore(sess, "./value_black.ckpt")
    saver1.restore(sess, "./value_white.ckpt")
    V1 = tf.get_default_graph().get_tensor_by_name("value_network0/softmax:0")
    V2 = tf.get_default_graph().get_tensor_by_name("value_network1/softmax:0")

    [d1, w1, wp1, pos1, d2, w2, wp2, pos2] = game.play_games(V1, [],\
            r1, V2, [], r2, n_train, nargout = 8)
    
    # Add the first moves to the data sets
    empty_boards =\
    np.concatenate((np.zeros((5,5,2,n_train)),np.ones((5,5,1,n_train))),axis=2)
    d2 = np.concatenate((empty_boards,d2[:,:,:,:]), axis = 3)
    w2 = np.concatenate((w1[0:n_train],w2), axis = 0)
    pos2 = np.concatenate((d1[:,:,0,0:n_train],pos2), axis = 2)

    d = [d1, d2]
    w = [w1, w2]
    pos = [pos1, pos2]
    
    for i in range(2):

        # Data augmentation
        print "Data augmentation step ..."
        [d[i], w[i], pos[i]] = data_augmentation(d[i], w[i], pos[i])
        d[i] = np.rollaxis(d[i], 3)
        pos[i] = np.rollaxis(pos[i],2).reshape((len(w[i]),game.nx*game.ny))
        print "Done!"

        data_index = np.arange(len(w[i]))
        shuffle(data_index)
        iteration = 0
        
        # Train next generation value and policy networks.
        print "training for policy network ", 1 - i
        for epoch in range(max_epoch):
            num_batch = np.ceil(len(data_index) / float(size_minibatch))
            for batch_index in range(int(num_batch)):
                batch_start = batch_index * size_minibatch
                batch_end = \
                        min((batch_index + 1) * size_minibatch, len(w[i]))
                indices = data_index[nrange(batch_start, batch_end)]
                pscope = "policy_network" + str(1 - i)
                
                # Extract good data from win games
                temp_indices = [idx for idx in indices if w[i][idx] == 2-i]
                
                # Remove invalid data for policy net
                filtered_indices = [idx for idx in temp_indices if\
                        np.amin(pos[i][idx,:])!=np.amax(pos[i][idx,:])]
                feed_dict = {\
                        pscope+"/S:0": d[i][filtered_indices, :, :, :],\
                        pscope+"/Y_:0": pos[i][filtered_indices, :]}
                
                sess.run(Optimizer[1 - i], feed_dict = feed_dict)
                iteration += 1
                if iteration % 10 == 0:
                    print "Epoch: ", epoch,\
                            "\t| Iteration: ", iteration,\
                            "\t| Loss: ",\
                            sess.run(Loss[1 - i], feed_dict = feed_dict)
        
        # Save the variables of the current policy network
        if i == 1:
            saver2.save(sess,"./SL_policy_black0.ckpt")
            print "Save SL_policy_black0.ckpt"
            P1 = tf.get_default_graph().get_tensor_by_name("policy_network0/softmax:0")
        else:
            saver3.save(sess,"./SL_policy_white0.ckpt")
            print "Save SL_policy_white0.ckpt"
            P2 = tf.get_default_graph().get_tensor_by_name("policy_network1/softmax:0")
    
    print "Evaluating neural network"

    r_all = np.ones((n_test)) # random moves for all games
    r_none = np.zeros((n_test)) # deterministic moves for all games
    
    s = game.play_games([], P1, r_none, [], [], r_all, n_test, nargout = 1)
    win[0]= s[0][0]; loss[0]= s[0][1]; tie[0] = s[0][2];
    print " net plays black: win=", win[0],\
            "\tloss=", loss[0],\
            "\ttie=", tie[0]
    s = game.play_games([], [], r_all, [], P2, r_none, n_test, nargout = 1)
    win[1] = s[0][1]; loss[1] = s[0][0]; tie[1] = s[0][2];
    print " net plays white: win=", win[1],\
            "\tloss=", loss[1],\
            "\ttie=", tie[1]
