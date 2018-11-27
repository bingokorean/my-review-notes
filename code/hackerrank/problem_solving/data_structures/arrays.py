####################################################
""" New Year Chaos """
####################################################
''' by Shafaet, Medium, 40, https://www.hackerrank.com/challenges/new-year-chaos/problem?h_l=interview&isFullScreen=false&page=10&playlist_slugs%5B%5D%5B%5D%5B%5D%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D%5B%5D%5B%5D%5B%5D=arrays
다음과 같은 배열을
[1, 2, 5, 3, 4, 7, 8, 6]
초기 배열 [1,2,3,4,5,6,7,8]로 만들자.
어떻게? 인접한 숫자와 바꾸면서.. (바꿀 때는 항상 앞사람하고 바꾸는 것을 기억하자!)
최소한으로 총 바꾼 횟수를 출력하자.
단, 현재 숫자는 인접 숫자에게 바꾸자는 제안을 최대 2번까지 할 수 있다.
만약, 3번 이상이 된다면, "Too chaotic"을 출력하자.
### Sample Input
2
5
2 1 5 3 4
5
2 5 1 3 4 
### Sample Output
3
Too chaotic
'''
#!/bin/python
import math
import os
import random
import re
import sys

# reference: Madoca
def minimumBribes(q, n):
    b = {} # 바꾼 횟수를 저장하기 위한 해쉬맵.
    r = 0
    cont = True
    
    while cont:
        cont = False
        for i in xrange(n-1):
            if q[i] > q[i+1]:
                if not q[i] in b:
                    b[q[i]] = 0
                b[q[i]] += 1
                if b[q[i]] > 2:
                    cont = False
                    r = "Too chaotic"
                    break
                q[i], q[i+1] = q[i+1], q[i]
                r += 1
                cont = True
    print(r)    
    
if __name__ == '__main__':
    t = int(raw_input())
    for t_itr in xrange(t):
        n = int(raw_input())
        q = map(int, raw_input().rstrip().split())
        minimumBribes(q, n)


####################################################
""" 2D Array - DS """
####################################################
''' by Shafaet, Easy, 15
Given, 6 x 6 2D array,
1 1 1 0 0 0
0 1 0 0 0 0
1 1 1 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0
and with all-one filter, which move their space with stride 1.
1 1 1
  1
1 1 1
find maximum weighted sum value

< Sample Input >
1 1 1 0 0 0
0 1 0 0 0 0
1 1 1 0 0 0
0 0 2 4 4 0
0 0 0 2 0 0
0 0 1 2 4 0
< Sample Output >
19
'''

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the hourglassSum function below.
def hourglassSum(arr):
    sum_list = []
    
    #print(arr[0][0], arr[1][0], arr[2][0])
    #print(arr[0][1], arr[1][1])
    
    
    for i, _ in enumerate(arr):
        if i==4: break
        
        for j, val in enumerate(_):
            if j==4: break
            sum_temp = 0
            
            sum_temp = arr[i][j] + arr[i][j+1] + arr[i][j+2] + \
                                    arr[i+1][j+1] + \
                        arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2] 
            
            sum_list.append(sum_temp)
     
    #print(sum_list)
    return max(sum_list)
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))
    result = hourglassSum(arr)
    fptr.write(str(result) + '\n')
    fptr.close()

    
    
    
    
####################################################
""" Arrays: Left Rotation """
####################################################
''' by Heraldo, Easy, 20
< Sample Input >
5 4
1 2 3 4 5
< Sample Output >
5 1 2 3 4
< Explanation >
When we perform d=4 left rotations, the array undergoes the following sequence of changes:
[1,2,3,4,5]->[2,3,4,5,1]->[3,4,5,1,2]->[4,5,1,2,3]->[5,1,2,3,4]

'''

#!/bin/python3
import math
import os
import random
import re
import sys

# Complete the rotLeft function below.
def rotLeft(a, d):
    return a[d:] + a[:d]
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nd = input().split()
    n = int(nd[0])
    d = int(nd[1])
    a = list(map(int, input().rstrip().split()))
    result = rotLeft(a, d)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()







    
