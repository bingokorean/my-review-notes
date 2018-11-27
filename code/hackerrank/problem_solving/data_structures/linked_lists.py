#################################################################
""" Insert a node at a specific position in a linked list """
#################################################################
''' by harsha_s, Easy, 5
Given a linked list, insert data with position

< Sample Input >
3
16
13
7
1
2
< Sample Output >
16 13 1 7
< Explanation >
The initial linked list is 16 13 7. We have to insert 1 at the position 2 which currently has 7 in it. 
The updated linked list will be 16 13 1 7
'''

#!/bin/python3
import math
import os
import random
import re
import sys

class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)

        if not self.head:
            self.head = node
        else:
            self.tail.next = node


        self.tail = node

def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))

        node = node.next

        if node:
            fptr.write(sep)

			
# Complete the insertNodeAtPosition function below.
def insertNodeAtPosition(head, data, position):
    
    #print(head.data)
    #print(head.next)
    
    cur_node = head
    node = SinglyLinkedListNode(data)
    
    i = 0
    while True:

        if i == position-1:
            temp_cur_next = cur_node.next
            cur_node.next = node
            
        elif i == position:
            cur_node.next = temp_cur_next
            break
        
        cur_node = cur_node.next
        i += 1
    
    return head

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()

    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)

    data = int(input())
    position = int(input())
    llist_head = insertNodeAtPosition(llist.head, data, position)
    print_singly_linked_list(llist_head, ' ', fptr)
    fptr.write('\n')
    fptr.close()
    
    
    
