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
game = game2()

def hyperparameters():
    global alpha, size_minibatch, max_epoch, n_train_list

    # Initial Learning Rate
    alpha = 0.0025
    # size of minibatch
    size_minibatch = 1024
    # training epoch
    max_epoch = 10
    # number of training steps for each generation
    n_train_list = [10000, 25000]

hyperparameters()
#####################################################################
"""                COMPUTATIONAL GRAPH CONSTRUCTION               """
#####################################################################

### DEFINE OPTIMIZER ###
def network_optimizer(Y, Y_, alpha, scope):
    # Cross entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
    # Parameters in this scope
    variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = scope)
    # L2 regularization
    for i in range(len(variables)):
        loss += 0.0001 * tf.nn.l2_loss(variables[i])
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(alpha).minimize(loss,\
            var_list = variables)
    return loss, optimizer


### NETWORK ARCHITECTURE ###
def network(state, nx, ny):
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
   
    # Create variable named "weights1fc" and "biases1fc".
    weights1fc = tf.get_variable("weights1fc", [nx*ny*50, 100], initializer = init)
    biases1fc = tf.get_variable("biases1fc", [100], initializer = init)
    
    # Create 1st fully connected layer
    fc1 = tf.reshape(out2, [-1, nx*ny*50])
    out1fc = tf.nn.relu(tf.matmul(fc1, weights1fc) + biases1fc)

    # Create variable named "weights2fc" and "biases2fc".
    weights2fc = tf.get_variable("weights2fc", [100, 3], initializer = init)
    biases2fc = tf.get_variable("biases2fc", [3], initializer = init)

    # Create 2nd fully connected layer
    return tf.matmul(out1fc, weights2fc) + biases2fc

# Define value network.
scope = "value_network"

with tf.variable_scope(scope):
    # Input
    S = tf.placeholder(tf.float32, shape = [None, game.nx, game.ny, 3], name = "S")
    # Estimation for unnormalized log probability
    Y = network(S, game.nx, game.ny) 
    # Estimation for probability
    P = tf.nn.softmax(Y, name = "softmax")
    # Target in integer
    W = tf.placeholder(tf.int32, shape = [None], name = "W")
    # Target in one-hot vector
    Y_= tf.one_hot(W, 3, name = "Y_")
    # Define loss and optimizer for value network
    loss, optimizer = network_optimizer(Y, Y_, alpha, scope)

### SAVER ###
saver = tf.train.Saver()

#####################################################################
"""                 TRAINING AND TESTING NETWORK                  """
#####################################################################

with tf.Session() as sess:
    ### DEFAULT SESSION ###
    sess.as_default()

    win1 = []; lose1 =[]; tie1 =[];
    win2 = []; lose2 =[]; tie2 =[];
 
    ### VARIABLE INITIALIZATION ###
    sess.run(tf.initialize_all_variables())
    
    for generation in range(len(n_train_list)):
        
        print "Generation: ", generation
                
        if generation == 0:
            # number of games to play for training
            n_train = n_train_list[generation] 
            # number of games to play for testing
            n_test = 10000
            # randomness for all games
            r1 = np.ones((n_train)) # randomness for player 1 for all games
            r2 = np.ones((n_train)) # randomness for player 2 for all games
            [d, w] = game.play_games([], [], r1, [], [], r2, n_train, nargout = 2)
        else:
            # Play 'ng' games between two players using the previous
            # generation value network 
            # introduce randomness in moves for robustness
            n_train = n_train_list[generation] # number of games to play for training
            n_test = 10000 # number of games to play for testing
            mt = np.floor(game.nx * game.ny / 2)
            r1r = rand(n_train, 1)
            r2r = rand(n_train, 1)
            r1k = randi(mt * 2, size = (n_train, 1))
            r2k = randi(mt * 2, size = (n_train, 1))
            r1 = (r1k > mt) * r1r + (r1k <= mt) * (-r1k)
            r2 = (r2k > mt) * r2r + (r2k <= mt) * (-r2k)
    
            [d, w] = game.play_games(P, [], r1, P, [], r2, n_train, nargout = 2)

        # Data augmentation
        print "Data augmentation step ..."
        [d, w, _] = data_augmentation(d, w, [])
        d = np.rollaxis(d, 3)
        print "Done!"
        
        data_index = np.arange(len(w))
        shuffle(data_index) 
        iteration = 0
          
        # Train the next generation value network
        for epoch in range(max_epoch):
            num_batch = np.ceil(len(data_index) / float(size_minibatch))
            for batch_index in range(int(num_batch)):
                batch_start = batch_index * size_minibatch
                batch_end = \
                        min((batch_index + 1) * size_minibatch, len(w))
                indices = data_index[nrange(batch_start, batch_end)]
                feed_dict = {S: d[indices, :, :, :], W: w[indices]}
                sess.run(optimizer, feed_dict = feed_dict)
                iteration += 1
                if iteration % 50 == 0:
                    print "Epoch: ", epoch,\
                            "\t| Iteration: ", iteration,\
                            "\t| Loss: ", sess.run(loss, feed_dict = feed_dict)

        # Save session.
        saver.save(sess, "./tictactoe_generation_"+str(generation)+".ckpt")
        # Load session
        # saver.restore(sess, "./tictactoe_generation_"+str(generation)+".ckpt")

        print "Evaluating generation ", generation, "neural network"
    
        r1 = np.zeros((n_test)) # randomness for player 1 for all games
        r2 = np.ones((n_test)) # randomness for player 2 for all games
        s = game.play_games(P, [], r1, [], [], r2, n_test, nargout = 1)
        win1.append(s[0][0]); lose1.append(s[0][1]); tie1.append(s[0][2]);
        print " net plays black: win=", win1[generation],\
                "\tlose=", lose1[generation],\
                "\ttie=", tie1[generation]
    
        r1 = np.ones((n_test)) # randomness for player 1 for all games
        r2 = np.zeros((n_test)) # randomness for player 2 for all games
        s = game.play_games([], [], r1, P, [], r2, n_test, nargout = 1)
        win2.append(s[0][1]); lose2.append(s[0][0]); tie2.append(s[0][2]);
        print " net plays white: win=", win2[generation],\
                "\tlose=", lose2[generation],\
                "\ttie=", tie2[generation]
