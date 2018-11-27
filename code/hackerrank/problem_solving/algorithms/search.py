####################################################
""" Sherlock and Array """
####################################################
''' by darkshadows, Easy, 40
Watson gives Sherlock an array of integers. His challenge is to find an element of the array such that the sum of all elements to the left is equal to the sum of all elements to the right.

< Sample Input 0 >
2
3
1 2 3
4
1 2 3 3
< Sample Output 0 >
NO
YES
< Explanation 0 >
For the first test case, no such index exists. 
For the second test case, arr[0]+arr[1]=arr[3], therefore index 2 satisfies the given conditions.
'''
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the balancedSums function below.
'''
# 이렇게 매번 sum함수를, 그것도 2번이나 콜하면 time complexity가 높아짐.
def balancedSums(arr):

    for i, _ in enumerate(arr):
        if sum(arr[:i]) == sum(arr[i+1:]):
            return 'YES'
    return 'NO'
''' 
    
# 이렇게 sum함수는 처음에 딱 1번만 콜하고, 나머지는 산수연산으로 해결함.
# 이렇게 하면 time complexity를 많이 줄일 수 있음.
def balancedSums(arr):    
    
    sum_all = sum(arr)
    sum_temp = 0
    
    for i, _ in enumerate(arr):    
        sum_temp += arr[i]
        sum_left = sum_temp - arr[i]
        sum_right = sum_all - sum_temp 
    
        #print(sum_left, sum_right)
        if sum_left == sum_right:
            return 'YES'
    
    return 'NO'
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    T = int(input().strip())
    for T_itr in range(T):
        n = int(input().strip())
        arr = list(map(int, input().rstrip().split()))
        result = balancedSums(arr)
        fptr.write(result + '\n')
    fptr.close()

	
	
####################################################
""" Hash Tables: Ice Cream Parlor """
####################################################
''' by dheeraj, Medium, 35
there are n=5 flavors having cost=[2,1,3,5,6]. Together they have money=5 to spend. They would purchase flavor ID's 1 and 3 for a cost of 2+3=5. Use 1 based indexing for your response.

< Sample Input >
2
4
5
1 4 5 3 2
4
4
2 2 4 3
< Sample Output >
1 4
1 2
< Explanation >
Sunny and Johnny make the following two trips to the parlor:
    The first time, they pool together money=4 dollars. There are five flavors available that day and flavors 1 and 4 have a total cost of 1+3=4.
    The second time, they pool together money=4 dollars. There are four flavors available that day and flavors 1 and 2 have a total cost of 2+2=4.
''' 
#!/bin/python3
import math
import os
import random
import re
import sys

# Complete the whatFlavors function below.
# 예상하게도 가장 무식한 방법이므로 time-out 문제가 발생.
'''
def whatFlavors(cost, money):
    # 모든 가능한 pair에 접근하기 위한 알고리즘임.
    i=0
    while True:
        if i==len(cost)-1: break
        j=0
        if cost[i] >= money: # 최대한 탐색하는 경우의 수를 줄이기 위해
            i += 1
            continue
        while True:
            if i+j+1==len(cost)-1: break
            if cost[i+j+1] >= money: # 최대한 탐색하는 경우의 수를 줄이기 위해
                j += 1
                continue
            pair_sum = cost[i] + cost[i+j+1]
            if money == pair_sum:
                print('{} {}'.format(i+1, i+j+1+1))
                return None
            j += 1
        i += 1
'''
def whatFlavors(cost, money):
    prices = dict()
    for i, _ in enumerate(cost):
        if money - cost[i] in prices: # 핵심.
            return prices[ money - cost[i] ], i
        prices[ cost[i] ] = i
        #print(prices)
    return None    

if __name__ == '__main__':
    t = int(input())
    for t_itr in range(t):
        money = int(input())
        n = int(input())
        cost = list(map(int, input().rstrip().split()))
        #whatFlavors(cost, money)
        f1, f2 = whatFlavors(cost, money)
        print(f1+1, f2+1)    	
	