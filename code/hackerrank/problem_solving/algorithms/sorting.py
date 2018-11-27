####################################################
""" Big Sorting """
####################################################
""" by _mfv_, Easy, 20
< Sample Input 0 >
6
31415926535897932384626433832795
1
3
10
3
5
< Sample Output 0 >
1
3
3
5
10
31415926535897932384626433832795
"""

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the bigSorting function below.
"""
# (단순) 버블정렬은 이 Big Sorting에서 timeout이 발생함.
def bubbleSort(unsorted):
    sorted_ = []
    while True:
        for i, x in enumerate(unsorted):
            if i==len(unsorted)-1:
                continue
            if int(unsorted[i]) > int(unsorted[i+1]):
                temp = unsorted[i]
                unsorted[i] = unsorted[i+1]
                unsorted[i+1] = temp
        sorted_.append(unsorted[-1])
        #print(unsorted)
        unsorted = unsorted[:-1]
        #print(unsorted)
        if len(unsorted) == 1:
            sorted_.append(unsorted[-1])
            break
    return sorted_[::-1] # sorted list
"""

"""
# (효율적인) 퀵정렬도 역시 이 Big Sorting에서 timeout이 발생함.
def quickSort(global_unsorted_, start_idx, end_idx):
    
    #print('hi', start_idx, end_idx)
    #print(global_unsorted_)
    
    subset_unsorted = global_unsorted_[start_idx:end_idx+1]
    
    if end_idx - start_idx == 1: return 0
    
    left, right = [], []
    for i, _ in enumerate(subset_unsorted): # end_idx+1-1 이유: 맨 뒤에 있는 pivot은 traverse안 함.
        if i == len(subset_unsorted)-1: continue
        if int(subset_unsorted[-1]) >= int(subset_unsorted[i]): # pivot index = -1
            left.append(subset_unsorted[i])
        else:
            right.append(subset_unsorted[i])
        
    global_unsorted_[start_idx:end_idx+1] = left + [subset_unsorted[-1]] + right
    
    quickSort(global_unsorted_, 0, len(left)-1)
    quickSort(global_unsorted_, len(left)-1, len(left)-1+1+len(right))

"""

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    '''
    n = int(input())

    unsorted = []

    for _ in range(n):
        unsorted_item = input()
        unsorted.append(unsorted_item)

    #result = bubbleSort(unsorted)
    #quickSort(unsorted, 0, len(unsorted)-1)
    #result = unsorted
 
    fptr.write('\n'.join(result))
    fptr.write('\n')

    fptr.close()
    '''
    
    # 문제는 sort함수를 만드는 것이 아니라 라이브러리에서 사용하는 것이고.
    # sorting의 기준을 수치에만 두는 것이 아니라 length에도 두는 것이다. 
    
    n = int(input().strip())
    unsorted = []
    unsorted_i = 0
    for unsorted_i in range(n):
       unsorted_t = str(input().strip())
       unsorted.append(unsorted_t)
    # your code goes here
    print(*sorted(unsorted, key=lambda x: (len(x), x)), sep='\n')    
    
    
    
    
    
####################################################
""" Sorting: Bubble Sort """
####################################################
''' by AvmnuSng, Easy, 30
Given an array of integers, sort the array in ascending order using the Bubble Sort algorithm above. Once sorted, print the number of swap, first instance, and last instance of the sorted array.
''' 
#!/bin/python3
import math
import os
import random
import re
import sys

# Complete the countSwaps function below.
def countSwaps(a):

    cnt = 0
    i=0
    while True:
        if i==len(a)-1: break
        j=0
        while True:
            if j==len(a)-1-i: break
            if a[j] > a[j+1]:
                cnt += 1
                temp = a[j]
                a[j] = a[j+1]
                a[j+1] = temp
            j+=1
            #print(a)
        i+=1
        
    print('Array is sorted in {} swaps.'.format(cnt))
    print('First Element: {}'.format(a[0]))
    print('Last Element: {}'.format(a[-1]))
    
if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().rstrip().split()))
    countSwaps(a)
    
    
    
