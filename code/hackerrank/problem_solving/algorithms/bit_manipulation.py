########################################################################
""" Lonely Integer """
########################################################################
''' by dheeraj, Easy, 20
You will be given an array of integers. All of the integers except one occur twice. That one is unique in the array. Find that unique integer.

< Sample Input 2 >
5
0 0 1 2 1
< Sample Output 2 >
2

Explanation 2
We have two 0's, two 1's, and one 2. 2 is unique.
'''

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the lonelyinteger function below.
'''
# 이렇게 풀어도 되지만 bit 연산이 아님.
def lonelyinteger(a):
    #print(a)
    for i, _ in enumerate(a):
        if a[i] == -1:
            continue
        for j, _ in enumerate(a):
            if i!=j:
                if a[i] == a[j]:
                    a[i] = -1
                    a[j] = -1
                    break    
    return [x for x in a if x!=-1][0]
'''
# XOR 비트 연산으로 효율적으로 문제를 풀 수 있음.
# 문제 자체가 XOR 비트 연산을 위한 문제임. 1번 나오는 숫자를 찾아라.
def lonelyinteger(a):
    answer = 0
    for i in a:
        answer = answer ^ i # XOR 비트 연산.
        print(answer)
    return answer


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    a = list(map(int, input().rstrip().split()))
    result = lonelyinteger(a)
    fptr.write(str(result) + '\n')
    fptr.close()