# EE488C Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016
# Information Theory & Machine Learning Lab, School of EE, KAIST
# Wonseok Jeon and Sungik Choi (wonsjeon@kaist.ac.kr, si_choi@kaist.ac.kr)
# written on 2016/11/23 

import numpy as np

def environment_maze(S, A, M):
    # Given the state S, action A, and map M, determine reward, new_state, and terminal
    # Map is represented as a length-40 vector that represents the connectivity
    #   between state (0->1)(1->2)....(23->24),(0->5)(1->6)(2->7)....(19->24):
    #   1 means we can go through them and 0 means the tranition is blocked
    # Action: 0: Up, 1: Down, 2, Left 3: Right
    T = 0
    reward = 0
    k1 = int(S / 5)  # row
    k2 = S - 5 * k1  # column
    new_state = S
    if ((k1 == 4) and (A == 0)) or ((k1 == 0) and (A == 1)) or\
            ((k2 == 0) and (A == 2)) or ((k2 == 4) and (A == 3)):
        # invalid move
        new_state = S
    elif A == 0:  # Goes up
        if M[S + 20] == 0:  # if not blocked
            new_state = S + 5
    elif A == 1:
        if M[S + 20 - 5] == 0:
            new_state = S - 5
    elif A == 2:
        if M[4 * k1 + k2 - 1] == 0:
            new_state = S - 1
    elif A == 3:
        if M[4 * k1 + k2] == 0:
            new_state = S + 1
    if new_state == 24:
        reward = 1
        T = 1
    return reward, new_state, T


def environment_maze_task2(S, A, M):
    # Given the state S, action A, and map M, determine reward, new_state, and terminal
    # Map is represented as a length-40 vector that represents the connectivity
    #   between state (0->1)(1->2)....(23->24),(0->5)(1->6)(2->7)....(19->24):
    #   1 means we can go through them and 0 means the tranition is blocked
    # Action: 0: Up, 1: Down, 2, Left 3: Right
    T = 0
    reward = 0
    k1 = int(S / 5)  # row
    k2 = S - 5 * k1  # column
    new_state = S
    if ((k1 == 4) and (A == 0)) or ((k1 == 0) and (A == 1)) or\
            ((k2 == 0) and (A == 2)) or ((k2 == 4) and (A == 3)):
        # invalid move
        new_state = S
        
    elif A == 0:  # Goes up
        if M[S + 20] == 0:  # if not blocked
            new_state = S + 5
    elif A == 1:
        if M[S + 20 - 5] == 0:
            new_state = S - 5
    elif A == 2:
        if M[4 * k1 + k2 - 1] == 0:
            new_state = S - 1
    elif A == 3:
        if M[4 * k1 + k2] == 0:
            new_state = S + 1
            
            
    if new_state == 4: # terminal state
        reward = 1
        T = 1 # terminal state arrivial !!!
    elif new_state == 6 or new_state == 7 or new_state == 8:
        reward = -100
        
        
    return reward, new_state, T





























