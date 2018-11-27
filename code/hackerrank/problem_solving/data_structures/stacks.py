####################################################
""" Balanced Brackets """
####################################################
'''
Check balanced brackets.
A matching pair of brackets is not balanced if the set of brackets it encloses are not matched.

< Sample Input >
3
{[()]}
{[(])}
{{[[(())]]}}
< Sample Output >
YES
NO
YES
'''
#!/bin/python3
import math
import os
import random
import re
import sys

'''
# Complete the isBalanced function below.
# 21문제 중 1문제는 실패. 14문제는 time-over. 왜지?
def isBalanced(s):
    
    s = list(s)
    right_bracket = [')', ']', '}']
    bracket_dict = {'(': ')', '[': ']', '{': '}'}
    stack_ = []
    
    if len(s) % 2 == 1:
        return 'NO'
    
    if s[0] in right_bracket:
        return 'NO'
    else:
        stack_.append(s[0])
    
    for i, _ in enumerate(s):
        if i==0: continue
        
        if s[i] in right_bracket:
            if s[i] == bracket_dict[stack_[-1]]:
                stack_.pop()
            else:
                return 'NO'
        else:
            stack_.append(s[i])
    return 'YES'
'''    

class Stack:
    def __init__(self):
        self.items = []
        
    def push(self,x):
        self.items.append(x)
        
    def pop(self):
        return self.items.pop()
    
    def last(self):
        return self.items[len(self.items)-1]
    
    def size(self):
        return len(self.items)
    
    def output(self):
        print(self.items)
    
def isBalanced(x):
    s=Stack()
    if len(x) % 2 == 1:
        return "NO"
    else:
        for i in range(len(x)):
            if x[i] == "{" or x[i] == "(" or x[i] == "[":
                s.push(x[i])
            elif s.size() == 0:
                break
            elif x[i] == "}" and s.last() == "{" :
                s.pop()
            elif x[i] == "]" and s.last() == "[" :
                s.pop()
            elif x[i] == ")" and s.last() == "(" :
                s.pop()
            else:
                break
        if i == len(x)-1 and s.size() == 0:
            return "YES"
        else:
            return "NO"
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        s = input()
        result = isBalanced(s)
        fptr.write(result + '\n')
    fptr.close()
