##################################################################################################################
""" The Coin Change Problem """
##################################################################################################################
''' by alanl, Medium, 60, https://www.hackerrank.com/challenges/coin-change/problem
값어치 기준으로 여러 종류의 코인이 있다. 종류는 주어지지만 개수를 무한대로 사용할 수 있다.
어떤 값이 주어지는데 이와 동일한 가능한 모든 경우의 코인 조합 개수를 세어보시오.
예) 4 types of coin and the value of each type is given 8,3,1,2 and the total value is 3.
이 경우 {1,1,1}, {1,2}, {3} 으로 총 3가지 경우의 수가 있다. 따라서 3을 출력한다 
(단, 조합 사이의 순서는 무시해도 된다)

### Sample Input
10 4
2 5 3 6
### Sample Output
5

1. {2,2,2,2,2}
2. {2,2,3,3}
3. {2,2,6}
4. {2,3,5}
5. {5,5}

'''
#!/bin/python
# thanks to - https://www.youtube.com/watch?v=PafJOaMzstY
import math
import os
import random
import re
import sys

'''
#####################################
### 첫 번째 시도 : recursive function을 사용해서 했으나, 짧은 입력의 테스트는 통과하지만 긴 입력은 시간 초과로 통과하지 않았다.
### recursive 함수로 dynamic programming하는 것은 굉장히 비효율적인 것 같다. 어찌됬든 결과값은 나온다.
global glob_cnt # recursive 함수의 return 횟수를 카운팅하기 위해 전역 변수를 따로 뒀다.
glob_cnt = 0
def recursive(c_list, bef_sum, n_sum, bef_idx):
    global glob_cnt
    for i, cur_c in enumerate(c_list):
        if bef_idx > i: # recursive가 한 단계 깊어짐에 있어서 index가 항상 정주행되도록.
            #print('~~~~', bef_idx, i, '(list)', c_list)
            continue
        cur_sum = bef_sum + cur_c 
        #print('->', cur_c)
        if cur_sum == n_sum:
            #print('---cur_sum:', cur_sum)
            glob_cnt += 1
            return 0
        elif cur_sum > n_sum:
            #print('+++cur_sum:', cur_sum)
            pass # 여기선 return하면 안된다. 다음 for문의 element들까지 search해야 하기 때문.
            #return 0
        elif cur_sum < n_sum:
            recursive(c_list, cur_sum, n_sum, i) # sum에 도달하지 않으면, 계속 재귀함수로 들어감.
        #return 0
        #filter_i += 1
        #n_jump += i
        #if i != len(c_list)-1:
        #    recursive(c_list, 0, n_sum, i) # sub recursive 
def getWays(n, c):
    n_sum = n
    c_list = c
    #for cur_c in c_list:
        #print(cur_c)
    #    recursive(c_list, cur_c, n_sum) # bef_sum -> cur_c
        #print('---------------------')
    #recursive(c_list, 0, n_sum, 0)
    for i, cur_c in enumerate(c_list):
        #print('Base--->', c_list[i])
        recursive(c_list[i:], c_list[i], n_sum, i-1) # c_list[i:]로 인해 i이 증가될 수록 list크기는 작아짐. 
        #break
#####################################
'''

# Complete the getWays function below.
def getWays(total, coin_arr):
    row_range = len(coin_arr)+1
    col_range = total+1
    coin_arr.insert(0, 0) # same as row_range by inserting zero index
    
    # cache array for dynamic programming
    cache_2d = [[0 for col in range(col_range)] for row in range(row_range)]
    
    # initial point (like recursion) - for bottom-up process
    for row_idx in range(row_range):
        cache_2d[row_idx][0] = 1

    ###
    ### Dynamic programming 
    for row_idx in range(row_range):
        if row_idx == 0: continue 
        for col_idx in range(col_range):
            #print(row_idx, col_idx)
            if col_idx == 0: continue
                
            ### Subproblem1. exclude current coin
            sub_prob1 = cache_2d[row_idx-1][col_idx]
            
            ### Subproblem2. include current coin
            # not using row_idx, but using coin_arr[row_idx]
            if col_idx - coin_arr[row_idx] < 0: # row is not index, but coin value
                sub_prob2 = 0
            else:
                sub_prob2 = cache_2d[row_idx][col_idx - coin_arr[row_idx]]
                
            ### Merge subproblems.
            cache_2d[row_idx][col_idx] = sub_prob1 + sub_prob2
            #print(cache_arr[row_idx])
    return (cache_2d[-1][-1])
    
if __name__ == '__main__':
    #fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nm = raw_input().split()
    n = int(nm[0])
    m = int(nm[1])
    c = map(long, raw_input().rstrip().split())
    ways = getWays(n, c)
    print(int(ways))
    #fptr.close()



#######################################################################
""" Max Array Sum """
#######################################################################
''' by shashank21j, Medium, 20, https://www.hackerrank.com/challenges/max-array-sum/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming
Given an array of integers, find the subset of non-adjacent elements with the maximum sum. Calculate the sum of that subset.
int형의 list array를 받아서, 인접하지 않는 element들로만 구성된 subset들을 모두 구하고 각각 sum을 구해라.
예를 들어, arr = [-2,1,3,-4,5]가 주어질 경우 다음과 같은 subset들이 구성된다
Subset      Sum
[-2, 3, 5]   6
[-2, 3]      1
[-2, -4]    -6
[-2, 5]      3
[1, -4]     -3
[1, 5]       6
[3, 5]       8
출력은 sum 중에서 가장 큰 값인 8을 출력하면 된다
만약 여기서 가장 큰 값을 출력하는 문제가 아니라면 dynamic programming을 할 수 있을까?

### Sample Input
5
3 7 4 6 5
### Sample Output
13
'''
#!/bin/python3
import math
import os
import random
import re
import sys

'''
# Complete the maxSubsetSum function below.
# recursion 버전은 속도가 엄청 느리다.
def recursion(sum_list, temp_sum, sub_arr):
    #print(sub_arr, len(sub_arr))
    #print(sum_list)
    if len(sub_arr)==2 or len(sub_arr)==1:
        #print('return!')
        return None
    for j, _ in enumerate(sub_arr):
        backup_sum = temp_sum
        if j==0 or j==1: continue # j==0: self, j==1: adjacent element
        backup_sum += sub_arr[j]
        #print(temp_sum)
        sum_list.append(backup_sum)
        recursion(sum_list, backup_sum, sub_arr[j:])
    #print('end return')
def maxSubsetSum(arr):
    sum_list = []
    for i, _ in enumerate(arr):
        if i==len(arr)-2: break
        recursion(sum_list, arr[i], arr[i:]) # start: arr[i]
    return max(sum_list)
'''

def maxSubsetSum(arr):
    cache = [0 for x in range(len(arr))]
    cache[0] = max(0, arr[0])
    cache[1] = max(cache[0], arr[1])
    if len(arr)==1:
        return arr[0]
    
    for i, _ in enumerate(arr):
        if i==0 or i==1: continue
        cache[i] = max( cache[i-2], max( (cache[i-2]+arr[i]), cache[i-1] ) ) # 음수때문에 max()를 하나 더.
    
    return max(cache[-2], cache[-1])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    res = maxSubsetSum(arr)
    fptr.write(str(res) + '\n')
    fptr.close()

