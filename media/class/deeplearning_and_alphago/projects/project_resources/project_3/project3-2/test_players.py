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
alpha_p = 1e-5 # learning rate for policy network
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
        #loss, optimizer = network_optimizer(Y, Y_, alpha_p, scope)
        # Append loss and optimizer.
        #Loss.append(loss); Optimizer.append(optimizer)
    
V0_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "value_network0")
V1_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "value_network1")
P0_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "policy_network0")
P1_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "policy_network1")

### SAVER ###
saver0 = tf.train.Saver(V0_variables, max_to_keep = 10000)
saver1 = tf.train.Saver(V1_variables, max_to_keep = 10000)
saver2 = tf.train.Saver(P0_variables, max_to_keep = 10000)
saver3 = tf.train.Saver(P1_variables, max_to_keep = 10000)

path = "./players/"
n_players = 9

with tf.Session() as sess:
    ### DEFAULT SESSION ###
    sess.as_default()

    win = np.zeros((n_players,n_players))
    loss = np.zeros((n_players,n_players))
    tie = np.zeros((n_players,n_players))
 
    ### VARIABLE INITIALIZATION ###
    sess.run(tf.initialize_all_variables())
    n_test = 1
    r_none = np.zeros((n_test))
    for i in range(n_players):
        for j in range(n_players):
            if i < 7 and j < 7:
                saver0.restore(sess, path+"black"+str(i+1)+".ckpt")
                saver1.restore(sess, path+"white"+str(j+1)+".ckpt")
                V0 = tf.get_default_graph().get_tensor_by_name("value_network0/softmax:0")
                V1 = tf.get_default_graph().get_tensor_by_name("value_network1/softmax:0")
                s = game.play_games(V0, [], r_none, V1, [], r_none, n_test, nargout = 1)
            elif i < 7 and j >= 7:
                saver0.restore(sess, path+"black"+str(i+1)+".ckpt")
                saver3.restore(sess, path+"white"+str(j+1)+".ckpt")
                V0 = tf.get_default_graph().get_tensor_by_name("value_network0/softmax:0")
                P1 = tf.get_default_graph().get_tensor_by_name("policy_network1/softmax:0")
                s = game.play_games(V0, [], r_none, [], P1, r_none,\
                        n_test, policy_type = 1, nargout = 1)
            elif i >= 7 and j < 7:
                saver2.restore(sess, path+"black"+str(i+1)+".ckpt")
                saver1.restore(sess, path+"white"+str(j+1)+".ckpt")
                P0 = tf.get_default_graph().get_tensor_by_name("policy_network0/softmax:0")
                V1 = tf.get_default_graph().get_tensor_by_name("value_network1/softmax:0")
                s = game.play_games([], P0, r_none, V1, [], r_none, n_test,\
                        policy_type = 1, nargout = 1)
            else:
                saver2.restore(sess, path+"black"+str(i+1)+".ckpt")
                saver3.restore(sess, path+"white"+str(j+1)+".ckpt")
                P0 = tf.get_default_graph().get_tensor_by_name("policy_network0/softmax:0")
                P1 = tf.get_default_graph().get_tensor_by_name("policy_network1/softmax:0")
                s = game.play_games([], P0, r_none, [], P1, r_none, n_test,\
                        policy_type = 1,  nargout = 1)
            win[i,j]=s[0][0]; loss[i,j]=s[0][1]; tie[i,j]=s[0][2]
            print " black by player "+str(i+1)+", white by player "+str(j+1)+": win=", win[i,j],\
                        "\tloss=", loss[i,j], "\ttie=", tie[i,j]

    f = open('./test_players.csv','w+')
    stat = np.zeros((n_players))
    msg = ","
    for j in range(n_players):
        msg = msg + str(j+1) + ","
    f.write(msg + "\n")
    for i in range(n_players):
        msg = str(i+1) + ","
        for j in range(n_players):
            if win[i,j]:
                msg = msg + "b,"
                stat[i] += 1
                stat[j] -= 1
            elif loss[i,j]:
                msg = msg + "w,"
                stat[i] -= 1
                stat[j] += 1
            else:
                msg = msg + "tie,"
        f.write(msg + "\n")
    msg = ","
    for j in range(n_players):
        msg = msg + str(j+1) + ","
    f.write(msg + "\n")
    msg = ","
    for j in range(n_players):
        msg = msg + str(stat[j]) + ","
    f.write(msg + "\n")
    f.close()

