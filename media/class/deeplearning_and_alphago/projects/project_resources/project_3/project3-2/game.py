# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
#
# Revision history
# Originally written in Matlab by Sae-Young Chung in Apr. 2016
#   for EE405C Electronics Design Lab <Network of Smart Systems>, Spring 2016
# Python & TensorFlow porting by Wonseok Jeon, Narae Ryu and Jinhak Kim,
# Hwehee Chung, Sungik Choi in Nov. 2016
#   for EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016

import Tkinter as tk
import tkMessageBox
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d as conv2d
from numpy.random import rand
from numpy.random import randint as randi
from numpy import arange as nrange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from strategy import next_move
import time

def data_augmentation(d, w, pos):
    # data augmentation
    # original, horizontal flip, vertical flip, and both
    # if the board is square, additionally 90 degree rotation
    [nx, ny, nc, ng] = d.shape
    if nx == ny:
        f = 8
    else:
        f = 4
    ng_new = ng * f
    dnew = np.zeros((nx, ny, nc, ng_new))
    wnew = np.zeros((ng_new))
    posnew = np.zeros((nx, ny, ng_new))
    dnew[:, :, :, nrange(ng)] = d[:, :, :, :]
    dnew[:, :, :, nrange(ng) + 1 * ng] = d[::-1, :, :, :]
    dnew[:, :, :, nrange(ng) + 2 * ng] = d[:, ::-1, :, :]
    dnew[:, :, :, nrange(ng) + 3 * ng] = d[::-1, ::-1, :, :]
    if np.sum(pos):
        posnew[:, :, nrange(ng)] = pos[:, :, :]
        posnew[:, :, nrange(ng) + 1 * ng] = pos[::-1, :, :]
        posnew[:, :, nrange(ng) + 2 * ng] = pos[:, ::-1, :]
        posnew[:, :, nrange(ng) + 3 * ng] = pos[::-1, ::-1, :]

    if f==8:
        dnew[:, :, :, nrange(ng) + 4 * ng] = np.rollaxis(d, 1, 0)
        dnew[:, :, :, nrange(ng) + 5 * ng] = dnew[::-1, :, :, nrange(ng) + 4 * ng]
        dnew[:, :, :, nrange(ng) + 6 * ng] = dnew[:, ::-1, :, nrange(ng) + 4 * ng]
        dnew[:, :, :, nrange(ng) + 7 * ng] =\
                dnew[::-1, ::-1, :, nrange(ng) + 4 * ng]
        if np.sum(pos):
            posnew[:, :, nrange(ng) + 4 * ng] = np.rollaxis(pos, 1, 0)
            posnew[:, :, nrange(ng) + 5 * ng] = posnew[::-1, :, nrange(ng) + 4 * ng]
            posnew[:, :, nrange(ng) + 6 * ng] = posnew[:, ::-1, nrange(ng) + 4 * ng]
            posnew[:, :, nrange(ng) + 7 * ng] =\
                    posnew[::-1, ::-1, nrange(ng) + 4 * ng]

    for k in range(f):
        wnew[k * ng + nrange(ng)]=w

    return dnew, wnew, posnew

class common(next_move):
    def __init__(self):
        self.xm = 0
        self.e = 0
        self.ym = 0
        self.pb = 0
        self.pb2 = 0
        self.pb3 = 1

    def play_games(self, netV1, netP1, r1, netV2, netP2, r2, ng, policy_type = 0, max_time = 0, nargout = 1):
        # plays 'ng' games between two players
        # optional parameter: max_time (the number of moves per game), nargout (the number of output of play_games)
        # returns dataset and labels
        "Inputs"
        # self: game parameters
        # netV1: value network playing black. can be empty
        # netP1: policy network playing black. can be empty (if both netV1 and netP1 are empty, 'r1' should be 1) 
        # r1: randomness in the move, 0: no randomness, 1: pure random
        #   if r1<0, then the first |r1| moves are random
        # netV2: value network playing white. can be empty
        # netP2: policy network playing white. can be empty (if both netV2 and netP2 are empty, 'r2' should be 1)
        # r2: randomness in the move, 0: no randomness, 1: pure random
        #   if r2<0, then the first |r2| moves are random
        # ng: number of games to play
        # policy_type: 0 if action is taken based on the output probabilities of the policy network
        #           or 1 if greedy action is taken for policy network 
        "Return values"
        #   stat=play_games(netV1,netP1,r1,netV2,netP2,r2,ng): statistics for netV1 and netP1, stat=[win loss tie]
        #   [d,w,wp,stat]=play_games(netV1,netP1,r1,netV2,netP2,r2,ng,nargout=2,3, or 4)
        #     d: 4-d matrix of size nx*ny*3*nb containing all moves, where nb is the total number of board configurations
        #     w: nb*1, 0: tie, 1: black wins, 2: white wins
        #     wp (if nargout>=3):  win probabilities for the current player
        #     stat (if nargout==4): statistics for netV1 and netP1, stat=[win loss tie], for netV2 and netP1, swap win & loss
        #   [d_black,w_black,wp_black,d_white,w_white,wp_white,stat]=play_games(netV1,netP1,r1,netV2,netP2,r2,ng,nargout=6,7)
        #     d_black: 4-d matrix of size nx*ny*3*nb1 containing all moves by black
        #     w_black: nb1*1, 0: tie, 1: black wins, 2: white wins
        #     wp_black: win probabilities for black
        #     d_white: 4-d matrix of size nx*ny*3*nb2 containing all moves by white
        #     w_white: nb2*1, 0: tie, 1: black wins, 2: white wins
        #     wp_white: win probabilities for white
        #     stat (if nargout==7): statistics for net1, stat=[win loss tie], for net2 swap win & loss
        
        
        # board size 
        nx = self.nx; ny = self.ny

        # maximum trials for each game
        if max_time <= 0:
            np0 = nx * ny * 2
        else:
            np0 = max_time

        # 'm' possible board configurations
        m = np0 * ng
        d = np.zeros((nx, ny, 3, m))
        pos = np.zeros((nx,ny,m))
        
        # Check whether tie(0)/black win(1)/white win(2) in all board configurations
        w = np.zeros((m))

        # winning probability: (no work for 1st generation)       
        wp = np.zeros((m))

        # Check whether the configurations are valid for training
        valid_data = np.zeros((m))

        # Check whether the configurations are played by player 1 or player 2
        turn = np.zeros((m))

        # number of valid moves in previous time step
        vm0 = np.ones((ng))

        try: # If "game_init" exists, 
            [b, state] = self.game_init(ng)
        except AttributeError: # otherwise, (in this case, state is dummy)
            b = np.zeros((nx, ny, ng))
            state = np.zeros((0, ng))

        # maximum winning probability for each game
        wp_max = np.zeros((ng))

        # For each time step, check whether game is in progress or not.
        game_in_progress = np.ones((ng))

        # First player: player 1 (black)
        p = 1

        for k in range(np0):
            if p == 1:
                if policy_type == 0:
                    if netP1 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val(b, state, game_in_progress, netV1, r1, p, k)
                    elif netV1 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.pol(b, state, game_in_progress, netP1, r1, p, k)
                    else:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val_and_pol(b, state, game_in_progress,\
                                netV1, netP1, netV2, netP2, r1, p, k)
                elif policy_type == 1:
                    if netP1 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val(b, state, game_in_progress, netV1, r1, p, k)
                    elif netV1 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.pol(b, state, game_in_progress, netP1, r1, p,\
                                k, policy_type)
                    else:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val_and_pol(b, state, game_in_progress,\
                                netV1, netP1, netV2, netP2, r1, p, k)
                else:
                    print "invalid policy_type"
            else:
                if policy_type == 0:
                    if netP2 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val(b, state, game_in_progress, netV2, r2, p, k)
                    elif netV2 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.pol(b, state, game_in_progress, netP2, r2, p, k)
                    else:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val_and_pol(b, state, game_in_progress,\
                                netV1, netP1, netV2, netP2, r2, p, k)
                elif policy_type == 1:
                    if netP2 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val(b, state, game_in_progress, netV2, r2, p, k)
                    elif netV2 == []:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.pol(b, state, game_in_progress, netP2, r2, p,\
                                k, policy_type)
                    else:
                        b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                            self.val_and_pol(b, state, game_in_progress,\
                            netV1, netP1, netV2, netP2, r2, p, k)
                else:
                    print "invalid policy_type"

            w0, end_game, _, _ = self.winner(b, state)
            idx = nrange(k * ng, (k + 1) * ng)
            d[:, :, 0, idx] = (b == 1)
            d[:, :, 1, idx] = (b == 2)
            d[:, :, 2, idx] = (b == 0)
            
            wp[idx] = wp_max
            valid_data[idx] = game_in_progress * (n_valid_moves > 0)
            
            # information about who's the current player
            turn[idx] = p
            if k>0:
                for i in range(ng):
                    if x_pos[i] >= 0:
                        pos[int(x_pos[i]),int(y_pos[i]),(k-1)*ng+i] = 1
            
            game_in_progress *=\
                    ((n_valid_moves > 0) * (end_game == 0) +\
                    ((vm0 + n_valid_moves) > 0) * (end_game == -1))

            # For tic-tac-toe, end_game == -1 doesn't occurs"""
            # if end_game==1, game ends
            # if end_game==0, game ends if no more move is possible for the current player
            # if end_game==-1, game ends if no moves are possible for both players

            number_of_games_in_progress = np.sum(game_in_progress)
            if number_of_games_in_progress == 0:
                break

            p = 3 - p
            vm0 = n_valid_moves[:]

        for k in range(np0):
            idx = nrange(k * ng, (k + 1) * ng)
            w[idx] = w0[:] # final winner

        # player 1's stat
        win = np.sum(w0 == 1) / float(ng)
        loss = np.sum(w0 == 2) / float(ng)
        tie = np.sum(w0 == 0) / float(ng)

        varargout = []

        if nargout >= 8:
            fv = np.where(valid_data * (turn == 1))[0]
            varargout.append(d[:, :, :, fv])
            # include only valid moves by black
            varargout.append(w[fv])
            varargout.append(wp[fv])
            varargout.append(pos[:,:,fv])
            fv = np.where(valid_data * (turn == 2))[0]
            varargout.append(d[:, :, :, fv])
            # include only valid moves by white
            varargout.append(w[fv])
            varargout.append(wp[fv])
            varargout.append(pos[:,:,fv])
            if nargout >= 9:
                varargout.append([win, loss, tie])
        
        elif nargout >= 6:
            fv = np.where(valid_data * (turn == 1))[0]
            varargout.append(d[:, :, :, fv])
            # include only valid moves by black
            varargout.append(w[fv])
            varargout.append(wp[fv])
            fv = np.where(valid_data * (turn == 2))[0]
            varargout.append(d[:, :, :, fv])
            # include only valid moves by white
            varargout.append(w[fv])
            varargout.append(wp[fv])
            if nargout >= 7:
                varargout.append([win, loss, tie])

        elif nargout >= 2:
            fv = np.where(valid_data)[0]
            varargout.append(d[:, :, :, fv])
            varargout.append(w[fv])
            if nargout >= 3:
                varargout.append(wp[fv])
            if nargout >= 4:
                varargout.append([win, loss, tie])
        else:
            varargout.append([win, loss, tie])
        return varargout
 
    def push_callback(self, event):
        self.pb = 1

    def push_callback2(self, event):
        self.pb2 = 1

    def push_callback3(self, event):
        self.pb3 = 1

    def mouse_click(self, event):
        if event.button == 1:
            xc = event.xdata
            yc = event.ydata
            if xc > 0.5 and xc < (self.nx + 0.5) and yc > 0.5 and yc < self.ny + 0.5:
                self.e = 1
                self.xm = round(xc)
                self.ym = round(yc)

    def figure_close(event):
        self.pb = 1
       
    def play_interactive(self, netV1, netP1, r1, netV2, netP2, r2):
        # interactive board game
        # Usage 1)
        # game.play_interactive([],[],0,[],[],0): human vs human
        #   self: game parameters
        #   netV1, netP1, netV2, netP2 must be empty set
        # Usage 2-1)
        # game.play_interactive(netV1,[],r1,[],[],0): computer(value network) vs human
        #   game: game parameters
        #   netV1: neural network for value, can not be empty
        #       if you want to play with the randomly moving opponent, then r1
        #       must be 1 
        #   netV1 can be [netV1_black, netV1_white], i.e., two separate neural
        #       networks depending on who goes first
        # Usage 2-2)
        # game.play_interactive([],netP1,r1,[],[],0): computer(policy network) vs human
        #   self: game parameters
        #   netP1: neural network for policy, can not be empty
        #       if you want to play with the randomly moving opponent, then r1
        #       must be 1 
        #   netP1 can be [netP1_black, netP1_white], i.e., two separate neural
        #       networks depending on who goes first
        # Usage 2-3)
        # game.play_interactive(netV1,netP1,r1,[],[],0): computer(value and policy network) vs human
        #   self: game parameters
        #   netV1 and netP1: neural network for value and policy, can not be empty
        #       if you want to play with the randomly moving opponent, then r1
        #       must be 1 
        #   netV1 can be [netV1_black, netV1_white], i.e., two separate neural
        #       networks depending on who goes first
        #   netP1 can be [netP1_black, netP1_white], i.e., two separate neural
        #       networks depending on who goes first
        # Usage 3)
        # game.play_interactive(netV1,netP1,r1,netV2,netP2,r2): computer vs computer
        # (one move at a time when mouse is clicked)
        #   self: game parameters
        #   netV1: first neural network for value, can be empty, i.e., [] (in
        #       this case player 1 plays with policy network if netP1 is not empty)
        #   netP1: first neural network for policy, can be empty, i.e., [] (in
        #       this case player 1 plays with value network if netV1 is not empty)
        #   @if both networks are empty, then see Usage 1
        #   r1: randomize parameter for net1
        #   netV2: second neural network for value, can be empty, i.e., [] (in
        #       this case player 2 plays with policy network if netP2 is not empty)
        #   netP2: second neural network for policy, can be empty, i.e., [] (in
        #       this case player 2 plays with value network if netV2 is not empty)
        #   @if both networks are empty, then see Usage 1
        #   r2: randomize parameter for net2
        #   likewise, net can be [net_black, net_white] for netV1, netP1,
        #       netV2, netP2, i.e., two separate neural networks depending on
        #       who goes first.
        plt.ion()
        e = 0
        root = tk.Tk()
        root.withdraw()

        def drawo(b, stone, x, y):
            if b[x, y, 0] != 0:
                r = 0
                return b, stone, r
            stone[x, y].set_facecolor('white')
            stone[x, y].set_visible(True)
            b[x, y, 0] = 2
            r = 2
            return b, stone, r

        def drawx(b, stone, x, y):
            if b[x, y, 0] != 0:
                r = 0
                return b, stone, r
            stone[x, y].set_facecolor('black')
            stone[x, y].set_visible(True)
            b[x, y, 0] = 1
            r = 1
            return b, stone, r

        def draw_board(stone, new_board):
            for xx in range(nx):
                for yy in range(ny):
                    if new_board[xx, yy, 0] == 1:
                        stone[xx, yy].set_facecolor('black')
                        stone[xx, yy].set_visible(True)
                    elif new_board[xx, yy, 0] == 2:
                        stone[xx, yy].set_facecolor('white')
                        stone[xx, yy].set_visible(True)
                    else:
                        stone[xx, yy].set_visible(False)
            b = new_board
            return b, stone

        def draw_newboard(ax, stone, txt, txt_winner):
            nx = self.nx
            ny = self.ny

            if (board_type == 'go') == True:
                rect = patches.Rectangle((0.5, 0.5), float(nx), float(ny), linewidth = 1,\
                        edgecolor = [255.0/255, 239.0/255, 173.0/255],\
                        facecolor = [255.0/255, 239.0/255, 173.0/255])
                ax.add_patch(rect)

                for kk in range(nx):
                    plt.plot([kk + 1.0, kk + 1.0], [1, ny], color = [0,0,0])

                for kk in range(ny):
                    plt.plot([1, nx], [kk + 1.0, kk + 1.0], color = [0,0,0])
            elif (board_type == 'basic') == True:
                ax.add_patch(patches.Rectangle((0.5, 0.5), float(nx), float(ny), linewidth = 1,\
                            edgecolor = [0, 0, 0], facecolor = board_color))

                for kk in range(nx+1):
                    plt.plot([float(kk) + 0.5, float(kk) + 0.5], [0.5, ny + 0.5], color = 'black')

                for kk in range(ny+1):
                    plt.plot([0.5, nx + 0.5], [kk + 0.5, kk + 0.5], color = 'black')
            else:
                for xx in range(nx):
                    for yy in range(ny):
                        if np.mod(xx+yy+1, 2) == 0:
                            ax.add_patch(patches.Rectangle((float(xx) + 0.5,\
                                    float(yy) + 0.5), 1.0, 1.0,\
                                    facecolor = [102.0/255, 68.0/255, 46.0/255],\
                                    edgecolor = 'black', linewidth=1))
                        else:
                            ax.add_patch(patches.Rectangle((float(xx) + 0.5,\
                                float(yy) + 0.5), 1.0, 1.0,\
                                facecolor = [247.0/255,236.0/255,202.0/255],\
                                edgecolor = 'black',linewidth=1))

            for xx in range(nx):
                for yy in range(ny):
                    stone[xx, yy] = patches.Circle((xx + 1.0, yy + 1.0), 0.4,\
                            facecolor = 'black',\
                            visible = False, zorder = 10)
                    txt[xx,yy] = plt.text(xx + 1.0, yy + 1.0, '',\
                            horizontalalignment = 'center',\
                            verticalalignment = 'center',\
                            fontsize = 30,\
                            color = 'blue', visible = False, zorder = 20)
                    ax.add_patch(stone[xx, yy])

            txt_winner = plt.text((nx + 1.0) / 2, (ny + 1.0) / 2, '',\
                    horizontalalignment='center', fontsize = 60,\
                    color = 'blue', visible = False)

            return ax, stone, txt, txt_winner

        def init_board(stone, txt):
            for xx in range(nx):
                for yy in range(ny):
                    stone[xx, yy].set_visible(False)
                    txt[xx, yy].set_visible(False)
            txt_winner.set_visible(False)

            return stone, txt

        def show_probabilities(txt, show_p, nx, ny, wp_all, ox, oy):
            if show_p:
                for xx in range(nx):
                    for yy in range(ny):
                        if xx == ox and yy == oy:
                            txt[xx, yy].set_visible(False)

                        if wp_all[xx, yy] >= 0:
                            txt[xx, yy].set_text(round(wp_all[xx, yy] * 100.0))
                            txt[xx, yy].set_visible(True)
                        else:
                            txt[xx, yy].set_visible(False)
            return txt

        def hide_probabilities(txt, nx, ny):
            for xx in range(nx):
                for yy in range(ny):
                    txt[xx, yy].set_visible(False)
            return txt

        nlevels = 1
        rw = []
        nx = self.nx
        ny = self.ny
        show_p = 1

        human_players = 0

        check_empty1 = [0, 0]

        check_empty2 = [0, 0]

        if isinstance(netV1, list):
            if len(netV1) == 0:
                check_empty1[0] = 1
                netV1= [netV1, netV1]
            elif len(netV1) == 1:
                netV1 = [netV1[0], netV1[0]]
            #else:
            #    if np.mod(nlevels, 2) == 0:
            #        # Tree search is not supported for this project.
            #        net1 = [net1[1], net1[0]]
        else:
            netV1 = [netV1, netV1]

        if isinstance(netV2, list):
            if len(netV2) == 0:
                check_empty2[0] = 1
                netV2= [netV2, netV2]
            elif len(netV2) == 1:
                netV2 = [netV2[0], netV2[0]]
        else:
            netV2 = [netV2, netV2]


        if isinstance(netP1, list):
            if len(netP1) == 0:
                check_empty1[1] = 1
                netP1= [netP1, netP1]
            elif len(netP1) == 1:
                netP1 = [netP1[0], netP1[0]]
        else:
            netP1 = [netP1, netP1]

        if isinstance(netP2, list):
            if len(netP2) == 0:
                check_empty2[1] = 1
                netP2= [netP2, netP2]
            elif len(netP2) == 1:
                netP2 = [netP2[0], netP2[0]]
        else:
            netP2 = [netP2, netP2]

        if sum(check_empty1) == 2:
            human_players += 1
            r1 = 1

        if sum(check_empty2) == 2:
            human_players += 1
            r2 = 1

        x = 0
        y = 0
        b = np.zeros((nx, ny, 1))
        game_in_progress = 0
        display_result = 0
        move = 0
        vm0 = 1
        stone = np.zeros((nx, ny), dtype = np.object)
        txt = np.zeros((nx, ny), dtype = np.object)
        txt_winner = []
        plot_handle = []
        board_type = 'go'
        try:
            if (self.theme == 'check') == True:
                board_type = 'check'
            elif (self.theme == 'basic') == True:
                board_type = 'basic'
                if min([nx,ny]) <= 3:
                    board_color = 'white'
                else:
                    board_color = [43.0/255, 123.0/255, 47.0/255]       
        except AttributeError:
            pass
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, nx + 0.5)
        ax.set_ylim(0, ny + 0.5)
        ax, stone, txt, txt_winner = draw_newboard(ax, stone, txt, txt_winner)
        cid = fig.canvas.mpl_connect('button_press_event', self.mouse_click)
        cid2 = fig.canvas.mpl_connect('close_event', self.figure_close)
        axexit = plt.axes([0.125,0.223,0.13,0.06])
        axwhite = plt.axes([0.125,0.163,0.13,0.06])
        axblack = plt.axes([0.125,0.103,0.13,0.06])
        ax.axis('equal')
        ax.axis('off')
        bexit = Button(axexit,'Exit')
        bexit.on_clicked(self.push_callback)
        bwhite = Button(axwhite, 'New(white)')
        bwhite.on_clicked(self.push_callback2)
        bblack = Button(axblack, 'New(black)')
        bblack.on_clicked(self.push_callback3)
        plt.show()
        plt.waitforbuttonpress(0.00000000001)
        while 1:
            if self.pb==1:
                plt.close()
                break
            if (self.pb2==1) | (self.pb3==1):
                stone,txt=init_board(stone,txt)
                b=np.zeros((nx,ny,1))
                turn=1
                if self.pb2==1:
                    my_color=2
                else:
                    my_color=1
                game_in_progress=1
                display_result=0
                move=1
                vm0=1
                self.e=0
                self.pb2=0
                self.pb3=0
                try:
                    b,state=self.game_init(1)
                    b,stone=draw_board(stone,b)
                except AttributeError:
                    b=np.zeros((nx,ny,1))
                    state=np.zeros((0,1))

            if game_in_progress and (human_players==2):
                if self.e:
                    self.e=0
                    x=self.xm
                    y=self.ym
                    isvalid,bn,new_state=self.valid(b,state,self.xy(nx*(x-1)+y-1),turn)
                    if isvalid:
                        w,end_game,_,_=self.winner(bn,new_state)
                        b,stone=draw_board(stone,bn)
                        turn=3-turn
                        move += 1
                        state=new_state
                        if end_game==1:
                            game_in_progress=0
                            display_result=1
                        elif self.number_of_valid_moves(b,state,turn)==0:
                            if end_game<0:
                                isvalid,bn,state=self.valid(b,state,np.array([[-1,-1]]),turn)
                                if turn==1:
                                    tkMessageBox.showwarning('Warning', 'Black must pass.')
                                else:
                                    tkMessageBox.showwarning('Warning', 'White must pass.')
                                turn=3-turn
                                move +=1
                                if self.number_of_valid_moves(b,state,turn)==0:
                                    game_in_progress=0
                                    display_result=1
                            else:
                                game_in_progress=0
                                display_result=1
            elif game_in_progress and (human_players==1):
                if turn!=my_color:
                    start_time = time.time()
                    if check_empty1 == [0, 1]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.val(b, state, game_in_progress, netV1[turn-1],\
                        np.array([r1]), turn, move)

                    elif check_empty1 == [1, 0]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.pol(b, state, game_in_progress, netP1[turn-1],\
                        np.array([r1]), turn, move, policy_type=1)

                    elif check_empty1 == [0, 0]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.val_and_pol(b, state, game_in_progress,\
                        netV1[turn-1], netP1[turn-1], netV1[2-turn], netP1[2-turn],\
                        np.array([r1]), turn, move)
                    end_time = time.time()
                    print end_time-start_time, "sec"

                    w,end_game,_,_=self.winner(b,new_board)
                    if n_valid_moves:
                        b,stone=draw_board(stone,new_board)
                        if check_empty1 != [1, 1]:
                            if r1 == 0:
                                txt=show_probabilities(txt,show_p,nx,ny,wp_all,ox,oy)
                        else:
                            txt=hide_probabilities(txt,nx,ny)

                    if end_game==1:
                        game_in_progress=0
                        display_result=1
                    elif end_game==0 and n_valid_moves==0:
                        game_in_progress=0
                        display_result=1
                    elif (end_game<0) and (n_valid_moves+vm0==0):
                        game_in_progress=0
                        display_result=1
                    elif n_valid_moves==0:
                        tkMessageBox.showwarning('Warning', 'I pass.')
                        turn=3-turn
                        move=move+1
                        state=new_state
                        vm0=n_valid_moves
                        if self.number_of_valid_moves(b,state,turn)==0:
                            game_in_progress=0
                            display_result=1
                    else:
                        turn=3-turn
                        move +=1
                        state=new_state
                        vm0=n_valid_moves
                        if self.number_of_valid_moves(b,state,turn)==0:
                            if end_game<0:
                                isvalid,bn,state=self.valid(b,state,np.array([[-1,-1]]),turn)
                                tkMessageBox.showwarning('Warning', 'You must pass.')
                                turn = 3-turn
                                move +=1
                                vm0=0
                            else:
                                game_in_progress=0
                                display_result=1
                elif self.e:
                    self.e=0
                    x=self.xm
                    y=self.ym
                    if turn==my_color:
                        isvalid,bn,new_state=self.valid(b,state,self.xy(nx*(x-1)+(y-1)),turn)
                        if isvalid:
                            b,stone=draw_board(stone,bn)
                            turn = 3-turn
                            move +=1
                            state=new_state
                            vm0=1
                            txt=hide_probabilities(txt,nx,ny)
            elif game_in_progress and self.e:
                self.e=0
                if turn==my_color:
                    start_time = time.time()
                    if check_empty1 == [0, 1]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.val(b, state, game_in_progress, netV1[turn-1],\
                        np.array([r1]), turn, move)

                    elif check_empty1 == [1, 0]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.pol(b, state, game_in_progress, netP1[turn-1],\
                        np.array([r1]), turn, move, policy_type=1)

                    elif check_empty1 == [0, 0]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.val_and_pol(b, state, game_in_progress,\
                        netV1[turn-1], netP1[turn-1], netV1[2-turn], netP1[2-turn],\
                        np.array([r1]), turn, move)
                    end_time = time.time()
                    print end_time-start_time, "sec"
                else:
                    start_time = time.time()
                    if check_empty2 == [0, 1]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.val(b, state, game_in_progress, netV2[turn-1],\
                        np.array([r2]), turn, move)

                    elif check_empty2 == [1, 0]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.pol(b, state, game_in_progress, netP2[turn-1],\
                        np.array([r2]), turn, move, policy_type=1)

                    elif check_empty2 == [0, 0]:
                        new_board, new_state ,n_valid_moves,\
                                wp_max, wp_all, ox, oy =\
                        self.val_and_pol(b, state, game_in_progress,\
                        netV2[turn-1], netP2[turn-1], netV2[2-turn], netP2[2-turn],\
                        np.array([r2]), turn, move)
                    end_time = time.time()
                    print end_time-start_time, "sec"
                w,end_game,_,_=self.winner(new_board,new_state)
                if n_valid_moves:
                    b,stone=draw_board(stone,new_board)
                    if turn==my_color:
                        if check_empty1 != [1, 1]:
                            if (r1==0):
                                txt=show_probabilities(txt,show_p,nx,ny,wp_all,ox,oy)
                        else:
                            txt=hide_probabilities(txt,nx,ny)
                    else:
                        if check_empty2 != [1, 1]:
                            if (r2==0):
                                txt=show_probabilities(txt,show_p,nx,ny,wp_all,ox,oy)
                        else:
                            txt=hide_probabilities(txt,nx,ny)
                turn=3-turn
                move +=1
                state=new_state
                if end_game==1:
                    game_in_progress=0
                    display_result=1
                elif self.number_of_valid_moves(b,state,turn)==0:
                    if end_game<0:
                        isvalid,bn,state=self.valid(b,state,np.array([[-1,-1]]),turn)
                        if turn==1:
                            print('Black must pass.') # C 2 U
                        else:
                            print('White must pass.') # C 2 U
                        turn = 3-turn
                        move +=1
                        if self.number_of_valid_moves(b,state,turn)==0:
                            game_in_progress=0
                            display_result=1
                    else:
                        game_in_progress=0
                        display_result=1
            if game_in_progress:
                w,end_game,_,_=self.winner(b,state)
                if end_game==1:
                    game_in_progress=0
                    display_result=1
            if display_result:
                w,_,_,_=self.winner(b,state)
                display_result=0
                if human_players == 1:
                    if w==3-my_color:
                        tkMessageBox.showwarning('Warning', 'I won.')
                    elif w==my_color:
                        tkMessageBox.showwarning('Warning', 'You won.')
                    else:
                        tkMessageBox.showwarning('Warning', 'Tie.')
                elif human_players == 2:
                    if w==1:
                        tkMessageBox.showwarning('Warning', 'Black won.')
                    elif w==2:
                        tkMessageBox.showwarning('Warning', 'White won.')
                    else:
                        tkMessageBox.showwarning('Warning', 'Tie.')
                else:
                    if w==1:
                        print('Black won')
                    elif w==2:
                        print('White won')
                    else:
                        print('Tie')
            plt.show()
            plt.waitforbuttonpress(0.00000000001)

    def number_of_valid_moves(self, b, state, p):
        nx, ny, ng = np.shape(b)
        nv = np.zeros((ng, 1))
        for x in range(nx):
            for y in range(ny):
                r, _, _ = self.valid(b, state, self.xy(nx * x + y), p)
                nv += r
        return nv

    def xy(self, k): # xy position
        try:
            n = len(k)
        except TypeError:
            n = 1
        ixy = np.zeros((n, 2))
        ixy[:, 0] = np.floor(k / float(self.ny))
        ixy[:, 1] = np.mod(k, float(self.ny))
        return ixy

class game1(common):
    def __init__(self, nx = 5, ny = 5, name = 'simple go'):
        super(game1, self).__init__()
        self.nx = nx
        self.ny = ny
        self.name = name
    def game_init(self, ng):
        # Initialize board for simple go game
        "Inputs"
        #   nx, ny: board size
        #   ng: number of boards
        "Return values"
        #   b: board
        #   state: state for b

        b = np.zeros([self.nx,self.ny,ng])
        state = -np.ones([2,ng])
        return b, state
    def winner(self, b, state):
        "Inputs"
        #   b: current board(s), 0: no stone, 1: black, 2: white
        #   state: state for b
        #   nargout: number of output arguments (See "Usage".)
        "Return values"
        #   [r, end_game, s1, s2] = winner(b, state, nargout = ?)
        #   r
        #       0: tie
        #       1: black wins
        #       2: white wins
        #       -- This is the current winner.
        #       -- This may not be the final winner.
        #   end_game (optional)
        #       1 : game ends
        #       0 : game ends if no more move is possible for the current player.
        #       -1: game ends if no move is possible for both players.
        #   for this game, this will be always zero
        #   s1 (optional)
        #       score for black
        #   s2 (optional)
        #       score for white
        ng = b.shape[2]
        nx = self.nx; ny = self.ny
        r = np.zeros((ng))
        s1 = np.zeros((ng))
        s2 = np.zeros((ng))
        f = [[0,1,0],[1,1,1],[0,1,0]]
        b_temp = np.zeros([nx,ny])
        for j in range(ng):
            b_temp = np.array(b[:,:,j])
            e = np.ones([nx+2, ny+2])
            e[1:nx+1,1:ny+1] = 1.0 * (b_temp==1)
            g = conv2d(e, f, mode = 'valid')
            s1[j] = np.sum((g==4) * (b_temp==0))
            e[1:nx+1,1:ny+1] = 1.0 * (b_temp==2)
            g = conv2d(e, f, mode = 'valid')
            s2[j] = np.sum((g==4) * (b_temp==0))
        r = (s1>s2) + 2 * (s2>s1)
        return r, -np.ones((ng)), s1, s2

    def valid(self, b, state, xy, p):
        # Check if the move (x,y) is valid.
        # See winner_simple_go for game rules
        "Inputs"
        #   b: current board(s), 0: no stone, 1: black, 2: white
        #   state: extra state(s) for game rules
        #   xy=[xs, ys]: new position(s) (xs and ys are scalar or vector)
        #   p: current plyaer, 1 or 2 (scalar or vector)
        "Return values"
        #   [r,new_board,new_state] = valid_simple_go(b,state,xs,ys,p)
        #   r
        #       1: valid
        #       0: invalid
        #   new_board (optional): update board
        #   new_state (optional): update state
        ng = b.shape[2]
        nx = self.nx; ny = self.ny

        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:,0]
            ys = np.ones((ng)) * xy[:,1]
        else:
            xs = xy[:,0]
            ys = xy[:,1]

        # whether position is valid or not in that game
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        new_board[:,:,:] = b[:,:,:] # copy by values
        new_state = -np.ones([2,ng])

        o = 3-p #opponent
        sx = nx + 2
        sy = ny + 2

        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])
            
            if x == -1 or y == -1:
                continue
            if x == state[0,j] and y == state[1,j]: # prohibited due to pai; ko
                continue
            b1 = np.zeros([nx,ny]);
            b1[:,:] = b[:,:,j]
            b2 = np.zeros(b1.shape)
            if b1[x,y] == 0:
                r[j] = 1;
                b1[x,y] = p
                [opponent_captured, xc, yc, b1] = self.check_captured4(x,y,o,b1)
                if opponent_captured == 0:
                    if self.check_captured(x,y,b1): # if suicide move?
                        r[j] = 0 # invalid
                        b1[x,y] = 0
                    b1 = (b1 * (b1>0)) + (p * (b1<0))
                    if r[j]:
                        e = np.zeros([nx+2,ny+2]) + p
                        e[1:nx+1,1:ny+1] = b1
                    # reducing own single-size territory is not allowed unless
                    # it prevents the opponent from playing there to capture stones
                        if e[x,y+1] == p and e[x+2,y+1] == p and e[x+1,y] == p\
                                and e[x+1,y+2] == p:
                            b2[:,:] = b1[:,:]
                            b1[x,y] = o
                            [n_captured_temp, xc_temp, yc_temp, b1] =\
                                self.check_captured4(x,y,p,b1)
                            b1[:,:] = b2[:,:]
                            if n_captured_temp > 0:
                                r[j] = 1
                                b1[x,y] = p
                            else:
                                r[j] = 0
                                b1[x,y] = 0
                elif opponent_captured == 1:
                    b2[:,:] = b1[:,:]
                    b1[xc,yc] = o
                    [n_captured_temp, xc_temp, yc_temp, b1] =\
                            self.check_captured4(xc, yc, p, b1)
                    b1[:,:] = b2[:,:]
                    if n_captured_temp == 1:
                        new_state[0,j] = xc
                        new_state[1,j] = yc
                new_board[:,:,j] = b1[:,:]
        return r, new_board, new_state

    def check_captured4(self, x, y, c, b1):
        nx = self.nx; ny = self.ny
        xc = x
        yc = y
        n_captured = 0
        if x > 0 and b1[x-1,y] == c:
            if self.check_captured(x-1, y, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                xc = x - 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        if y > 0 and b1[x,y-1] == c:
            if self.check_captured(x, y-1, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                yc = y - 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        if x < nx-1 and b1[x+1,y] == c:
            if self.check_captured(x+1, y, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                xc = x + 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        if y < ny-1 and b1[x,y+1] == c:
            if self.check_captured(x, y+1, b1):
                n_captured = n_captured + np.sum(b1 < 0)
                yc = y + 1
                b1 = b1 * (b1>0) + 0
            else:
                # revert
                b1 = (b1 * (b1>0)) + (c * (b1<0))
        return n_captured, xc, yc, b1

    def check_captured(self, x, y, b1):
        nx = self.nx; ny = self.ny
        captured = 1
        c = b1[x,y]
        b1[x,y] = -1
        if x > 0 and b1[x-1,y] == 0:
            captured = 0
            b1[x,y] = c
            return captured 
        if y > 0 and b1[x,y-1] == 0:
            captured = 0
            b1[x,y] = c
            return captured
        if x < nx-1 and b1[x+1,y] == 0:
            captured = 0
            b1[x,y] = c
            return captured
        if y < ny-1 and b1[x,y+1] == 0:
            captured = 0
            b1[x,y] = c
            return captured
        if x > 0:
            if b1[x-1,y] == c:
                captured = self.check_captured(x-1, y, b1)
            if not captured:
                b1[x,y] = c
                return captured
        if y > 0:
            if b1[x,y-1] == c:
                captured = self.check_captured(x, y-1, b1)
            if not captured:
                b1[x,y] = c
                return captured
        if x < nx-1:
            if b1[x+1,y] == c:
                captured = self.check_captured(x+1, y, b1)
            if not captured:
                b1[x,y] = c
                return captured
        if y < ny-1:
            if b1[x,y+1] == c:
                captured = self.check_captured(x, y+1, b1)
            if not captured:
                b1[x,y] = c
                return captured
        return captured


class game2(common):
    def __init__(self, nx = 3, ny = 3, n = 3, name = 'tic tac toe', theme = 'basic'):
        super(game2, self).__init__()
        self.nx = nx
        self.ny = ny
        self.n = n # n-mok
        self.name = name
        self.theme = theme

    def winner(self, b, state):
	# Check who wins for n-mok game
	"Inputs"
	#	self: game parameters
	#	b: current board(s), 0: no stone, 1: black, 2: white
	#	state: extr state for b
	# Usage) [r, end_game, s1, s2]=winner_n_mok(game,b)
	#	r: 0 tie, 1: black wins, 2: white wins
	#	end_game (optional)
	#		if end_game==1, game ends
	#		if end_game==0, game ends if no more move is possible for the current player
	#		if end_game==-1, game ends if no moves are possible for both players
	#	s1 (optional): score for black
	#	s2 (optional): score for white
        # total number of games
        ng = b.shape[2]
        n = self.n
        r = np.zeros((ng))
        fh = np.ones((n, 1))
        fv = np.transpose(fh)
        fl = np.identity(n)
        fr = np.fliplr(fl)
        for j in range(ng):
            c = (b[:, :, j] == 1)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 1

            c = (b[:, :, j] == 2)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 2
        return r, r > 0, r == 1, r == 2

    def valid(self, b, state, xy, p):
	# Check if the move (x,y) is valid for a basic game where any empty board position is possible.
	"Inputs"
	#	self: game parameters
	#	b: current board(s), 0: no stone, 1: black, 2: white
	#	state: state for b
	#	xy=[xs, ys]: new position(s) (xs and ys are scalar or vector)
	#	p: current player, 1 or 2 (scalar or vector)
	"Return values"
	#	[r,new_board,new_state]=valid_basic(game,b,state,xs,ys,p)
	#	r: 1 means valid, 0 means invalid
	#	new_board (optional): updated board
	#	new_state (optional): update state
        ng = b.shape[2]
        n = self.n
        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:, 0]
            ys = np.ones((ng)) * xy[:, 1]
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]

        # whether position is valid or not in that game
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :] # copy by values
        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])

            if x == -1 or y == -1:
                continue
            if b[x, y, j] == 0: # position is empty in the j-th game
                r[j] = 1 # check valid
                new_board[x, y, j] = p # check black or white in new board

        return r, new_board, state

class game3(common):
    def __init__(self, nx = 9, ny = 9, n = 5, name = '5-mok'):
        super(game3, self).__init__()
        self.nx = nx
        self.ny = ny
        self.n = n # n-mok
        self.name = name

    def winner(self, b, state):
	# Check who wins for n-mok game
	"Inputs"
	#	self: game parameters
	#	b: current board(s), 0: no stone, 1: black, 2: white
	#	state: extr state for b
	# Usage) [r, end_game, s1, s2]=winner_n_mok(game,b)
	#	r: 0 tie, 1: black wins, 2: white wins
	#	end_game (optional)
	#		if end_game==1, game ends
	#		if end_game==0, game ends if no more move is possible for the current player
	#		if end_game==-1, game ends if no moves are possible for both players
	#	s1 (optional): score for black
	#	s2 (optional): score for white
        # total number of games
        ng = b.shape[2]
        n = self.n
        r = np.zeros((ng))
        fh = np.ones((n, 1))
        fv = np.transpose(fh)
        fl = np.identity(n)
        fr = np.fliplr(fl)

        for j in range(ng):
            c = (b[:, :, j] == 1)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 1

            c = (b[:, :, j] == 2)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                    or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 2

        return r, r > 0, r == 1, r == 2

    def valid(self, b, state, xy, p):
	# Check if the move (x,y) is valid for a basic game where any empty board position is possible.
	"Inputs"
	#	self: game parameters
	#	b: current board(s), 0: no stone, 1: black, 2: white
	#	state: state for b
	#	xy=[xs, ys]: new position(s) (xs and ys are scalar or vector)
	#	p: current player, 1 or 2 (scalar or vector)
	"Return values"
	#	[r,new_board,new_state]=valid_basic(game,b,state,xs,ys,p)
	#	r: 1 means valid, 0 means invalid
	#	new_board (optional): updated board
	#	new_state (optional): update state
        ng = b.shape[2]
        n = self.n

        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:, 0]
            ys = np.ones((ng)) * xy[:, 1]
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]

        # whether position is valid or not in that game
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b # copy by values
        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])

            if x == -1 or y == -1:
                continue
            if b[x, y, j] == 0: # position is empty in the j-th game
                r[j] = 1 # check valid
                new_board[x, y, j] = p # check black or white in new board

        return r, new_board, state

class game4(common):
    def __init__(self, nx = 8, ny = 8, name = 'othello', theme = 'basic'):
        super(game4, self).__init__()
        # Initialize the class "game4".
        "Inputs"
        #   nx, ny: board size
        self.nx = nx
        self.ny = ny
        self.name = name
        self.theme = theme

    def game_init(self, ng):
        # Initialize board for simple go game
        "Inputs"
        #   nx, ny: board size
        #   ng: number of boards
        "Return values"
        #   b: board
        #   state: state for b
        if self.nx < 3 or self.ny < 3:
            raise Exception('Board is too small')

        b = np.zeros((self.nx, self.ny, ng))
        sx = int(np.floor(self.nx / 2) - 1)
        sy = int(np.floor(self.ny / 2) - 1)
        b[sx, sy, :] = 1
        b[sx + 1, sy, :] = 2
        b[sx, sy + 1, :] = 2
        b[sx + 1, sy + 1, :] = 1
        state = np.zeros((0, ng))

        return b, state

    def winner(self, b, state):
        # Check who wins for othello game.
        "Inputs"
        #   b: current board(s), 0: no stone, 1: black, 2: white
        #   state: extr state for b
        #   nargout: number of output arguments (See "Usage".) 
        "Return values"
        #   [r, end_game, s1, s2] = winner(b, state, nargout = ?)
        #   r
        #       0: tie,
        #       1: black wins
        #       2: white wins
        #       -- This is the current winner.
        #       -- This may not be the final winner.
        #   end_game (optional)
        #       1 : game ends
        #       0 : game ends if no more move is possible for the current player.
        #       -1: game ends if no move is possible for both players. 
        #   s1 (optional)
        #       score for black
        #   s2 (optional)
        #       score for white
        ng = b.shape[2]
        r = np.zeros((ng))
        end_game = np.zeros((ng, 1))

        s0 = np.squeeze((b == 0).sum(axis = 0, keepdims = True).sum(axis = 1,\
                keepdims = True))

        s1 = np.squeeze((b == 1).sum(axis = 0, keepdims = True).sum(axis = 1,\
                keepdims = True))

        s2 = np.squeeze((b == 2).sum(axis = 0, keepdims = True).sum(axis = 1,\
                keepdims = True))

        r = (s1 > s2) + (s2 > s1) * 2

        return r, 1 - 2 * (s0 > 0) * (s1 > 0) * (s2 > 0), \
                s1, s2
        # For the second argument,
        # returns 1, if the board is full or one of the players have no stones,
        # retures -1, otherwise.

    def valid(self, b, state, xy, p):
        # Check if the move (x,y) is valid for othello game.
        "Inputs"
        #   game: game parameters
        #   b: current board(s)
        #       0: no stone
        #       1: black
        #       2: white
        #   state: state for b
        #   xy = [xs, ys]: new position(s) (xs and ys are scalar or vector)
        #   p: current player
        #       1: scalar
        #       2: vector
        #   nargout: number of output arguments (See "Usage".) 
        "Return values"
        #   [r, new_board, new_state] = valid(b, state, xy, p, nargout = ?)
        #   r
        #       1: valid
        #       0: invalid
        #   new_board (optional): updated board
        #   new_state (optional): updated state
        ng = b.shape[2]

        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:, 0]
            ys = np.ones((ng)) * xy[:, 1]
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]

        # whether position is valid or not in that game
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        b1 = np.zeros((b.shape[0], b.shape[1]))

        new_board[:, :, :] = b[:, :, :]

        dx = np.array([1,  1,  0, -1, -1, -1,  0,  1])
        dy = np.array([0, -1, -1, -1,  0,  1,  1,  1])

        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])

            if x == -1 or y == -1:
                continue
            if b[x, y, j] == 0:
                b1[:, :] = b[:, :, j]
                v = 0
                for z in range(len(dx)):
                    if self.check_captured(x, y, dx[z], dy[z], p, b1):
                        v = 1
                        b1 = b1 * (b1 > 0) + p * (b1 < 0)
                    else:
                        b1 = b1 * (b1 > 0) + (3 - p) * (b1 < 0) # restore
                if v:
                    r[j] = 1
                    b1[x, y] = p
                    new_board[:, :, j] = b1[:, :]

        return r, new_board, state

    def check_captured(self, x, y, dx, dy, c, b1):
        r = 0
        o = 0
        for a in range(min(self.nx, self.ny)):
            x = x + dx
            y = y + dy
            if x >= 0 and x < self.nx and y>= 0 and y < self.ny:
                if b1[x, y] == c:
                    if o:
                        r = 1
                    return r
                elif b1[x, y] == 3 - c:
                    o = 1
                    b1[x, y] = -1
                else:
                    return r
            else:
                return r
