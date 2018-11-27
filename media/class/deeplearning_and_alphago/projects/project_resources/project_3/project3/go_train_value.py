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

#####################################################################
"""                    DEFINE HYPERPARAMETERS                     """
#####################################################################
# Choose game.
game = game1()
# Initial Learning Rate
alpha_v = 0.002 # learning rate for value network
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

### NETWORK CONSTRUCTION ###
"""
    value_network0 for black
    value_network1 for white
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
        # Define loss and optimizer for value network
        loss, optimizer = network_optimizer(Y, Y_, alpha_v, scope)
        # Append loss and optimizer.
        Loss.append(loss); Optimizer.append(optimizer)
   
V0_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "value_network0")
V1_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "value_network1")

### SAVER ###
saver0 = tf.train.Saver(V0_variables, max_to_keep = 10000)
saver1 = tf.train.Saver(V1_variables, max_to_keep = 10000)

with tf.Session() as sess:
    ### DEFAULT SESSION ###
    sess.as_default()

    win1 = []; lose1 =[]; tie1 =[];
    win2 = []; lose2 =[]; tie2 =[];
 
    ### VARIABLE INITIALIZATION ###
    sess.run(tf.initialize_all_variables())
    
    for generation in range(2):

        print "Generation: ", generation
        
        if generation == 0:
            # number of games to play for training
            n_train = 1000
            # number of games to play for testing
            n_test = 1000
            
            r_all = np.ones((n_train)) # random moves for all games
            
            [d1, w1, wp1, d2, w2, wp2] = game.play_games([], [],\
                    r_all, [], [], r_all, n_train, nargout = 6)
        else:
            # Play 'ng' games between two players using the previous
            # generation value network 
            # introduce randomness in moves for robustness
            n_train = 2500 # number of games to play for training
            n_test = 2500 # number of games to play for testing
            mt = np.floor(game.nx * game.ny / 2)
            r1r = rand(n_train, 1)
            r2r = rand(n_train, 1)
            r1k = randi(mt * 2, size = (n_train, 1))
            r2k = randi(mt * 2, size = (n_train, 1))
            r1 = (r1k > mt) * r1r + (r1k <= mt) * (-r1k)
            r2 = (r2k > mt) * r2r + (r2k <= mt) * (-r2k)
    
            [d1, w1, wp1, d2, w2, wp2] = game.play_games(V1, [], r1, V2, [],\
                    r2, n_train, nargout = 6)

        d = [d1, d2]
        w = [w1, w2]
        
        for i in range(2):

            # Data augmentation
            print "Data augmentation step ..."
            [d[i], w[i], _] = data_augmentation(d[i], w[i], [])
            d[i] = np.rollaxis(d[i], 3)
            print "Done!"

            data_index = np.arange(len(w[i]))
            shuffle(data_index)
            iteration = 0
            
            # Train next generation value networks.
            print "training for value network ", i
            for epoch in range(max_epoch):
                num_batch = np.ceil(len(data_index) / float(size_minibatch))
                for batch_index in range(int(num_batch)):
                    batch_start = batch_index * size_minibatch
                    batch_end = \
                            min((batch_index + 1) * size_minibatch, len(w[i]))
                    indices = data_index[nrange(batch_start, batch_end)]
                    vscope = "value_network" + str(i)
                    feed_dict = {\
                            vscope+"/S:0": d[i][indices, :, :, :],\
                            vscope+"/W:0": w[i][indices]}
                    sess.run(Optimizer[i], feed_dict = feed_dict)
                    iteration += 1
                    if iteration % 50 == 0:
                        print "Epoch: ", epoch,\
                                "\t| Iteration: ", iteration,\
                                "\t| Loss: ",\
                                sess.run(Loss[i], feed_dict = feed_dict)
        
            # Save the variables of the current value network
            if i == 0:
                saver0.save(sess,"./value_black_gen_"+str(generation)+".ckpt")
                print "Save value_black_gen_"+str(generation)+".ckpt"
                V1 = tf.get_default_graph().get_tensor_by_name("value_network0/softmax:0")
            else:
                saver1.save(sess,"./value_white_gen_"+str(generation)+".ckpt")
                print "Save value_white_gen_"+str(generation)+".ckpt"
                V2 = tf.get_default_graph().get_tensor_by_name("value_network1/softmax:0")
        
        print "Evaluating generation ", generation, "neural network"
        
        r_all = np.ones((n_test)) # random moves for all games
        r_none = np.zeros((n_test)) # deterministic moves for all games
    
        s = game.play_games(V1, [], r_none, [], [], r_all, n_test, nargout = 1)
        win1.append(s[0][0]); lose1.append(s[0][1]); tie1.append(s[0][2]);
        print " net plays black: win=", win1[generation],\
                "\tlose=", lose1[generation],\
                "\ttie=", tie1[generation]

        s = game.play_games([], [], r_all, V2, [], r_none, n_test, nargout = 1)
        win2.append(s[0][1]); lose2.append(s[0][0]); tie2.append(s[0][2]);
        print " net plays white: win=", win2[generation],\
                "\tlose=", lose2[generation],\
                "\ttie=", tie2[generation]
