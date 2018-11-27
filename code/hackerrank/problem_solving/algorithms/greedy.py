########################################################################
""" Candies """
########################################################################	
""" by HackerRank, Medium, 50, https://www.hackerrank.com/challenges/candies/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming
점수를 가진 학생들이 있다. 선생님은 학생에게 캔디를 나눠주려고 한다.
적어도 한 사람당 캔디 1개씩 나눠줘야 하고, 근접한 학생의 점수보다 높으면 그보다 더 많은 캔디를 줘야 한다.
어떻게 하면 최소한의 캔디를 나눠줄 수 있을까?
ex) 
student's ratings are [4,6,4,5,6,2]
studuents candy in minimal amounts: [1,2,1,2,3,1]

### Sample Input
10 <- 개수
2
4
2
6
1
7
8
9
2
1

### Sample Output
19  <- sum of optimal distribution 1,2,1,2,1,2,3,4,2,1
"""
# reference: bannr's code
#!/bin/python3
import math
import os
import random
import re
import sys

# Complete the candies function below.
def candies(n, rate_arr):

    idx_arr = list(range(n))
    candy_arr = [0] * n
    
    idx_arr.sort(key = lambda i: rate_arr[i]) # 여기서 정렬하는 로직이 핵심이다. 점수 크기에 맞게 idx가 오름차순 정렬된다.
    
    for i in idx_arr: # 점수가 낮은 학생들부터 시작한다.
        # 최소한 1개의 캔디를 제공해야 한다.
	# 양옆(left&right candy)을 모두 고려해서 가장 큰 값을 선택하는 구조이다.
	left_candy = 1 
        right_candy = 1
        
        # left candy
        if i: # first index pass 
            if rate_arr[i-1] < rate_arr[i]:
                left_candy = candy_arr[i-1] + 1    
        # right candy
        if i < n-1: # last index pass
            if rate_arr[i+1] < rate_arr[i]:
                right_candy = candy_arr[i+1] + 1
                
        candy_arr[i] = max(left_candy, right_candy)
    
    return sum(candy_arr)
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = []
    for _ in range(n):
        arr_item = int(input())
        arr.append(arr_item)
    result = candies(n, arr)
    fptr.write(str(result) + '\n')
    fptr.close()



####################################################
""" Minimum Absolute Difference in an Array """
####################################################
''' by shashank21j, Easy, 15
Given an array of integers, find and print the minimum absolute difference between any two elements in the array. 

< Sample Input 0 >
3
3 -7 0
< Sample Output 0 >
3
< Explanation 0 >
The smallest of these possible absolute differences is 3 ( |3 - 0| ).

< Sample Input 1 >
10
-59 -36 -13 1 -53 -92 -2 -96 -54 75
< Sample Output 1 >
1
< Explanation 1 >
The smallest absolute difference is 1 ( |-54 - - 53| ).
'''
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the minimumAbsoluteDifference function below.
'''
# 이중 for문은 time complexity에서 막힘.
def minimumAbsoluteDifference(arr):
    diff_list = []
    
    while True:
        #print(arr)
        if len(arr)==1:
            break
        for i, _ in enumerate(arr):
            if i==len(arr)-1:
                break
            diff_list.append(abs(arr[0]-arr[i+1]))
        arr = arr[1:]
        
    #print(diff_list)
    return min(diff_list)
'''
'''
# 이중 for문은 time complexity에서 막힘.
def minimumAbsoluteDifference(arr):
    diff_list = []
    
    i=0
    while True:
        if i==len(arr)-1: break
        j=1 # j=0이면 자기 자신과 비교함.
        while True:
            if j==len(arr)-i: break
            diff_list.append( abs(arr[i] - arr[i+j]) )
            j += 1
        i += 1
            
    return min(diff_list)
'''
    
# 상대적인 차이가 적은 것에 아이디어를 얻어야 한다.
# 가장 무식한 방법은 이중 for문을 돌려서 모든 경우의 조합을 다 계산하는 것이다.
# 그런데, sorting을 먼저 한 다음에는? for문 한 번만 돌리면 된다.
def minimumAbsoluteDifference(arr):    
    arr.sort()
    diff_list = []
    for i, _ in enumerate(arr):
        if i==len(arr)-1: break
        diff_list.append( abs(arr[i] - arr[i+1]) )
    return min(diff_list)
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    result = minimumAbsoluteDifference(arr)
    fptr.write(str(result) + '\n')
    fptr.close()
