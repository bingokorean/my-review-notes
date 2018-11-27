# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
#
# Revision history
# Originally written in Matlab by Sae-Young Chung in Apr. 2016
#   for EE405C Electronics Design Lab <Network of Smart Systems>, Spring 2016
# Python & TensorFlow porting by Wonseok Jeon, Narae Ryu, Hwehee Chung in Dec. 2016
#   for EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016

import numpy as np
import tensorflow as tf
from numpy.random import rand
from numpy.random import randint as randi
from numpy import arange as nrange

class next_move(object):
# returns next move by using neural networks
# this is a parallel version, i.e., returns next moves for multiple games
# Input arguments: self,b,state,game_in_progress,net,rn,p,move,nlevels,rw
# self: game parameters
# b: current board status for multiple games
# state: extra state for the board
# game_in_progress: 1 if game is in progress, 0 if ended
# net: neural network. can be empty (in that case 'rn' should be 1)
# rn: randomness in the move, 0: no randomness, 1: pure random
#   if rn<0, then the first |rn| moves are random
# p: current player (1: black, 2: white)
# move: k-th move (1,2,3,...)
# nlevels (optional): # of levels (1,2,3,...). default=1
#   if nlevels is even, then the opponent's neural network needs to be used
# rw (optional): randomization in calculating winning probabilities, default=0
# Return values
# [new_board,new_state,valid_moves,wp_max,wp_all,x,y]=next_move(b,game_in_progress,net,rn,p,move)
#   new_board: updated boards containing new moves
#   new_state: update states
#   n_valid_moves: number of valid moves
#   wp_max: best likelihood of winning
#   wp_all: (optional) likelihood of winning for all possible next moves
#   x: (optional) x coordinates of next moves
#   y: (optional) y coordinates of next moves

    def val(self, b, state, game_in_progress, net, rn, p, move, nlevels = 1, rw = 0):
        " nlevels: depth of tree search (<= 3)"
        " rw: randomness for checking winning probability"
        
        # board size
        nx = self.nx; ny = self.ny; nxy = nx * ny
        # randomness for each game & minimum r
        r = rn; rmin = np.amin(r)
        # number of games
        if b.ndim>=3:
            ng = b.shape[2]
        else:
            ng=1
        # number of valid moves in each game 
        n_valid_moves = np.zeros((ng))
        # check whether moves ('nxy' moves) are valid
        valid_moves = np.zeros((ng, nxy))
        # win probability for each position on this each game
        wp_all = np.zeros((nx, ny, ng))
        # maximum over wp_all
        wp_max = -np.ones((ng))
        mx = np.zeros((ng))
        my = np.zeros((ng))
        x = -np.ones((ng))
        y = -np.ones((ng))

        # check nlevels
        if nlevels > 3 or nlevels < 0:
            raise Exception('# of levels invalid. Should be 1,2, or 3.')
        # total cases to consider in tree search
        ncases = pow(nxy, nlevels)

        # All possible boards after 'b'
        d = np.zeros((nx, ny, 3, ng * ncases), dtype = np.int32)

        for p1 in range(nxy):
            vm1, b1, state1 = self.valid(b, state, self.xy(p1), p)
            n_valid_moves += vm1
            if rmin < 1:
                valid_moves[:, p1] = vm1
                if nlevels == 1:
                    idx = nrange(ng) + p1 * ng
                    d[:, :, 0, idx] = (b1 == 1)
                    d[:, :, 1, idx] = (b1 == 2)
                    d[:, :, 2, idx] = (b1 == 0)
                else:
                    for p2 in range(nxy):
                        vm2, b2, state2 = self.valid(b1, state1, self.xy(p2), 3 - p)
                        if nlevels == 2:
                            idx = nrange((ng)) + p1 * ng + p2 * ng * nxy
                            d[:, :, 0, idx] = (b1 == 1)
                            d[:, :, 1, idx] = (b1 == 2)
                            d[:, :, 2, idx] = (b1 == 0)
                        else:
                            for p3 in range(nxy):
                                vm3, b3, state3 = self.valid(b2, state2, self.xy(p3), p)
                                idx = nrange(ng) + p1 * ng + p2 * ng * nxy\
                                        + p3 * ng * nxy * nxy
                                d[:, :, 0, idx] = (b1 == 1)
                                d[:, :, 1, idx] = (b1 == 2)
                                d[:, :, 2, idx] = (b1 == 0)

        n_valid_moves = n_valid_moves * game_in_progress

        # For operations in TensorFlow, load session and graph
        sess = tf.get_default_session()

        # Axis rollaxis for placeholder inputs
        d = np.rollaxis(d, 3)
        if rmin < 1: # if not fully random
            softout = np.zeros((d.shape[0], 3))
            size_minibatch = 1000
            num_batch = np.ceil(d.shape[0] / float(size_minibatch))
            for batch_index in range(int(num_batch)):
                batch_start = batch_index * size_minibatch
                batch_end = \
                        min((batch_index + 1) * size_minibatch, d.shape[0])
                indices = range(batch_start, batch_end)
                try:
                    feed_dict = {"value_network"+str(p-1)+"/S:0": d[indices, :, :, :]}
                    softout[indices, :] = sess.run(net, feed_dict = feed_dict)
                except:
                    feed_dict = {"value_network"+"/S:0": d[indices, :, :, :]}
                    softout[indices, :] = sess.run(net, feed_dict = feed_dict)
            if p == 1:
                wp = 0.5 * (1 + softout[:, 1] - softout[:, 2])
            else:
                wp = 0.5 * (1 + softout[:, 2] - softout[:, 1])

            if rw != 0:
                wp = wp + np.random.rand((ng, 1)) * rw

            if nlevels >= 3:
                wp = np.reshape(wp, (ng, nxy, nxy, nxy))
                wp = np.amax(wp, axis = 3)

            if nlevels >= 2:
                wp = np.reshape(wp, (ng, nxy, nxy))
                wp = np.amin(wp, axis = 2)

            wp = np.transpose(np.reshape(wp,(nxy,ng)))
            wp = valid_moves * wp - (1 - valid_moves)
            wp_i = np.argmax(wp, axis = 1)
            mxy = self.xy(wp_i) # max position

            for p1 in range(nxy):
                pxy = self.xy(p1)
                wp_all[int(pxy[:, 0]), int(pxy[:, 1]), :] = wp[:, p1]

        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :]
        new_state = np.zeros(state.shape)
        new_state[:, :] = state[:, :]

        for k in range(ng):
            if n_valid_moves[k]: # if there are valid moves
                if (r[k] < 0 and np.ceil(move / 2) <= -r[k])\
                        or (r[k] >= 0 and rand() <= r[k]):
                # if moves for each player is less than |r[k]| for negative
                # r[k] or
                # random number is less than r[k] for positive r[k]
                # do any random action that is valid
                    while True:
                        # random position selection
                        rj = randi(nx)
                        rk = randi(ny)
                        rxy = np.array([[rj, rk]])
                        isvalid, _, _ =\
                                self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                        if int(isvalid[0]):
                            break

                    isvalid, bn, sn =\
                            self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                    new_board[:, :, [k]] = bn
                    new_state[:, [k]] = sn
                    x[k] = rj
                    y[k] = rk

                else:
                    isvalid, bn, sn =\
                            self.valid(b[:, :, [k]], state[:, [k]], mxy[[k], :], p)
                    new_board[:, :, [k]] = bn
                    new_state[:, [k]] = sn
                    x[k] = mxy[k, 0]
                    y[k] = mxy[k, 1]

            else: # if there is no valid moves
                isvalid, bn, sn =\
                        self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                new_state[:, [k]] = sn

        return new_board, new_state, n_valid_moves, wp_max,\
                wp_all, x, y

    def pol(self, b, state, game_in_progress, net, rn, p, move,\
            policy_type = 0, nlevels = 1, rw = 0):
        " nlevels: depth of tree search (<= 3)"
        " rw: randomness for checking winning probability"
        # board size
        nx = self.nx; ny = self.ny; nxy = nx * ny
        # randomness for each game & minimum r
        r = rn; rmin = np.amin(r)
        # number of games
        if b.ndim>=3:
            ng = b.shape[2]
        else:
            ng=1
        # number of valid moves in each game 
        n_valid_moves = np.zeros((ng))
        # check whether moves ('nxy' moves) are valid
        valid_moves = np.zeros((ng, nxy))
        # win probability for each position on this each game
        wp_all = np.zeros((nx, ny, ng))
        # maximum over wp_all
        wp_max = -np.ones((ng))
        mx = np.zeros((ng))
        my = np.zeros((ng))
        x = -np.ones((ng))
        y = -np.ones((ng))

        # check nlevels
        if nlevels > 1 or nlevels < 0:
            raise Exception('# of levels greater than 0 and less than 1 is supported.')

        # For operations in TensorFlow, load session and graph
        sess = tf.get_default_session()
        d = np.zeros((nx, ny, 3, ng))
        d[:, :, 0, :] = (b == 1)
        d[:, :, 1, :] = (b == 2)
        d[:, :, 2, :] = (b == 0)
        # Axis rollaxis for placeholder inputs
        d = np.rollaxis(d, 3)

        softout = np.zeros((d.shape[0], nx*ny))
        if rmin < 1: # if not fully random
            size_minibatch = 5000
            num_batch = np.ceil(d.shape[0] / float(size_minibatch))
            for batch_index in range(int(num_batch)):
                batch_start = batch_index * size_minibatch
                batch_end = \
                        min((batch_index + 1) * size_minibatch, d.shape[0])
                indices = range(batch_start, batch_end)
                feed_dict = {"policy_network"+str(p-1)+"/S:0": d[indices, :, :, :]}
                softout[indices, :] = sess.run(net, feed_dict = feed_dict)
            if rw != 0:
                wp = wp + np.random.rand((ng, 1)) * rw

        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :]
        new_state = np.zeros(state.shape)
        new_state[:, :] = state[:, :]

        
        for k in range(ng):
            if True: # if there are valid moves
                if (r[k] < 0 and np.ceil(move / 2) <= -r[k])\
                        or (r[k] >= 0 and rand() <= r[k]):
                # if moves for each player is less than |r[k]| for negative
                # r[k] or
                # random number is less than r[k] for positive r[k]
                # do any random action that is valid
                    temp_board = d[k, :, :, 0] + d[k, :, :, 1]
                    while True:
                        # random position selection
                        rj = randi(nx)
                        rk = randi(ny)
                        temp_board[rj, rk] = 1
                        rxy = np.array([[rj, rk]])
                        isvalid, _, _ =\
                                self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                        if int(isvalid[0]):
                            isvalid, bn, sn =\
                                self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                            new_board[:, :, [k]] = bn
                            new_state[:, [k]] = sn
                            x[k] = rj
                            y[k] = rk
                            n_valid_moves[k] += 1
                            break
                        if int(np.amin(temp_board)) != 0:
                            break
                    if not int(isvalid[0]):
                        isvalid, bn, sn =\
                            self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                        new_state[:, [k]] = sn

                elif policy_type == 0:
                    temp_softout = softout[k,:]/np.sum(softout[k,:])
                    while True:
                        cum_softout = np.cumsum(temp_softout)
                        softr = rand(1)
                        softxy = np.sum(softr>cum_softout)
                        rxy = np.array(self.xy(softxy))
                        rj = int(rxy[0][0])
                        rk = int(rxy[0][1])
                        isvalid, _, _ =\
                                self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                        if int(isvalid[0]):
                            isvalid, bn, sn =\
                                self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                            new_board[:, :, [k]] = bn
                            new_state[:, [k]] = sn
                            x[k] = rj
                            y[k] = rk
                            n_valid_moves[k] += 1
                            break
                        else:
                            temp_softout[softxy] = 0
                            norm = np.sum(temp_softout)
                            if norm != 0:
                                temp_softout = temp_softout/norm
                            else:
                                break
                    if not int(isvalid[0]):
                        isvalid, bn, sn =\
                            self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                        new_state[:, [k]] = sn
                else:
                    for num_traials in range(nxy):
                        max_xy = np.argmax(softout[k,:])
                        max_prob = softout[k,max_xy]
                        rxy = np.array(self.xy(max_xy))
                        softout[k,max_xy] = 0
                        isvalid, bn, sn =\
                                self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                        if int(isvalid[0]):
                            new_board[:, :, [k]] = bn
                            new_state[:, [k]] = sn
                            x[k] = rxy[:,0]
                            y[k] = rxy[:,1]
                            n_valid_moves[k] += 1
                            wp_max[k] = max_prob
                            break
                    if not int(isvalid[0]):
                        isvalid, bn, sn =\
                            self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                        new_state[:, [k]] = sn

        n_valid_moves = n_valid_moves * game_in_progress
        wp_all = np.reshape(softout, [ng, 5, 5])
        wp_all = np.rollaxis(np.rollaxis(wp_all, 2), 2)
        return new_board, new_state, n_valid_moves, wp_max,\
                wp_all, x, y

    def val_and_pol(self, b, state, game_in_progress, netV1, netP1, netV2, netP2,\
            rn, p, move, nlevels = 1, rw = 0):
        # write the code here below
        return
