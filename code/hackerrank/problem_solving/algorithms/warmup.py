#################################################
""" Solve Me First """
#################################################
''' by shashank21j (https://www.hackerrank.com/challenges/solve-me-first/problem)
Complete the function solveMeFirst to compute the sum of two integers.
# input
x = 2
y = 3
# output
5
The sum of the two integers and is computed as: 2 + 3 = 5
'''
def solveMeFirst(a,b):
   # Hint: Type return a+b below 
  return a+b

num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)


##################################################
""" Simple Array Sum """
##################################################
''' by shashank21j (https://www.hackerrank.com/challenges/simple-array-sum/problem)
Given an array of integers, find the sum of its elements.
# input
6
1 2 3 4 10 11
# output
31
We print the sum of the array's elements: 1 + 2 + 3 + 4 + 10 + 11 = 31
'''
import sys
'''
def simpleArraySum(n, ar):
    # Complete this function
    total = 0
    for idx, value in enumerate(ar):
        total += value
    return total
'''
def simpleArraySum(n, ar):
    return sum(ar)

n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = simpleArraySum(n, ar)
print(result)


##################################################
""" Compare the Triplets """
##################################################
''' by Shafaet (https://www.hackerrank.com/challenges/compare-the-triplets/problem)
(Alice and Bob)'s comparison points
# input
5 6 7
3 6 10
# output
1 1 
Alice's comparison score is 1 (5>3), and Bob's comparison score is 1 (7<10). 
'''
import sys

def const(num): # constraints
    if num in range(1,101): # 1 ~ 100
        return True
    else: 
        return False

def comp(a, b):
    if a > b:
        return 'big'
    elif a < b:
        return 'small'

def solve(a_arr, b_arr):
    # Complete this function
    alice, bob = 0, 0
    
    #for i, _ in enumerate(a_arr):
    #    if const(a_arr[i])==True and const(b_arr[i])==True:
    #        if comp(a_arr[i], b_arr[i])=='big':
    #            alice+=1
    #        elif comp(a_arr[i], b_arr[i])=='small':
    #            bob+=1

    for x,y in zip(a_arr,b_arr):            
        if const(x)==True and const(y)==True:
            if x > y:
                alice+=1
            elif x < y:
                bob+=1                

    return alice, bob

a0, a1, a2 = input().strip().split(' ')
a0, a1, a2 = [int(a0), int(a1), int(a2)]
b0, b1, b2 = input().strip().split(' ')
b0, b1, b2 = [int(b0), int(b1), int(b2)]
result = solve([a0, a1, a2], [b0, b1, b2])
print (" ".join(map(str, result)))


##################################################
""" A Very Big Sum """
##################################################
''' by vatsalchanana (https://www.hackerrank.com/challenges/a-very-big-sum/problem)
Calculate and print the sum of the elements in an array, keeping in mind that some of those integers may be quite large.
# input
5
1000000001 1000000002 1000000003 1000000004 1000000005
# output
5000000015
(참고) python에서는 long형을 따로 지정하지 않아도 됨. 
'''
import sys

def aVeryBigSum(n, ar):
    # Complete this function
    return sum(ar)
    
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = aVeryBigSum(n, ar)
print(result)


##################################################
""" Diagonal Difference """
##################################################
''' by vatsalchanana (https://www.hackerrank.com/challenges/diagonal-difference/problem)
Given a square matrix, calculate the absolute difference between the sums of its diagonals.
# input
3
11 2 4
4 5 6
10 8 -12
# output
15
Sum across the primary diagonal: 11 + 5 - 12 = 4
Sum across the secondary diagonal: 4 + 5 + 10 = 19 
Difference: |4 - 19| = 15
'''
import sys

def diagonalDifference(a):
    # Complete this function
    sum_prim_diag = 0
    sum_secd_diag = 0
    
    i=0
    for vec in a:
        temp = vec[::-1] # reverse a vector
        sum_prim_diag += vec[i]
        sum_secd_diag += temp[i] 
        i+=1
    
    return abs(sum_prim_diag-sum_secd_diag)

if __name__ == "__main__":
    n = int(input().strip())
    a = []
    for a_i in range(n):
       a_t = [int(a_temp) for a_temp in input().strip().split(' ')]
       a.append(a_t)
    result = diagonalDifference(a)
    print(result)


##################################################
""" Plus Minus """
##################################################
''' by vatsalchanana (https://www.hackerrank.com/challenges/plus-minus/problem)
Given an array of integers, calculate the fractions of its elements that are positive, negative, and are zeros. 
# input
6
-4 3 -9 0 4 1      
# output
0.500000
0.333333
0.166667
The proportions of occurrence are positive: 3/6=0.500000, negative: 2/6=0.333333 and zeros: 1/6=0.166667.
'''
import sys

def plusMinus(arr):
    # Complete this function
    total_n = len(arr)
    pos_n = 0
    neg_n = 0
    zero_n = 0
    for int in arr:
        if int > 0:
            pos_n += 1
        elif int < 0:
            neg_n += 1
        elif int == 0:
            zero_n += 1
    #print(round(pos_n/total_n, 6)) # 0.5
    #print(round(neg_n/total_n, 6)) # 0.333333
    #print(round(zero_n/total_n, 6)) # 0.166667
    
    print(format(pos_n/total_n, '.6f')) # 0.500000
    print(format(neg_n/total_n, '.6f')) # 0.333333
    print(format(zero_n/total_n, '.6f')) # 0.166667
    
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    plusMinus(arr)


##################################################
""" Staircase """
##################################################
''' by vatsalchanana (https://www.hackerrank.com/challenges/staircase/problem)
Consider a staircase of size and draw using # symbols and spaces. 
# input 
6
# output
     #
    ##
   ###
  ####
 #####
######
'''
import sys

def staircase(n):
    # Complete this function
    for i in reversed(range(n)):
        row = ''
        row += (' '*i)
        row += ('#'*(n-i))
        print(row)
    
if __name__ == "__main__":
    n = int(input().strip())
    staircase(n)


##################################################
""" Mini-Max Sum """
##################################################
''' by bishop15 (https://www.hackerrank.com/challenges/mini-max-sum/problem)
Given five positive integers, find the minimum and maximum values that can be calculated by summing exactly four of the five integers.
# input
1 2 3 4 5
# output
10 14
'''
import sys

def miniMaxSum(arr):
    # Complete this function
    total_sum = sum(arr)
    min_val = min(arr)
    max_val = max(arr)
    print(total_sum - max_val, total_sum - min_val)
    
if __name__ == "__main__":
    arr = list(map(int, input().strip().split(' ')))
    miniMaxSum(arr)

   ## implemented by delamath
   #ar = sorted(map(int, input().split()))
   #print(sum(ar[:-1]), sum(ar[1:]))

   
##################################################
""" Birthday Cake Candles """
##################################################
''' by shashank21j (https://www.hackerrank.com/challenges/birthday-cake-candles/problem)
When she blows out the candles, she’ll only be able to blow out the tallest ones.
# input
4
3 2 1 3
# output
2
Maximum value is 3. The count of 3 is 2.
'''
import sys

def birthdayCakeCandles(n, ar):
    # Complete this function
    #max_val = max(ar)
    #cnt = 0
    #for int in ar:
    #    if int==max_val:
    #        cnt+=1
    #return cnt
    
    return ar.count(max(ar))
 
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = birthdayCakeCandles(n, ar)
print(result)


##################################################
""" Time Conversion """
##################################################
''' by vatsalchanana (https://www.hackerrank.com/challenges/time-conversion/problem)
Given a time in 12-hour AM/PM format, convert it to military (24-hour) time.
# input
07:05:45PM
# output
19:05:45
'''
import sys

def timeConversion(s):
    # Complete this function
    listed_s = list(s)
    splited_s = s.split(':')
    hh = splited_s[0]
    mm = splited_s[1]
    ss = splited_s[2][:2]
    
    if listed_s[-2] == 'A':
        if s == '00:00:00AM':
            return '24:00:00'
        if int(hh) == 12:
            hh = '00'
        return hh+':'+mm+':'+ss
            
    elif listed_s[-2] == 'P':
        if int(hh) < 12:
            hh = int(hh) + 12
            hh = str(hh)
        return hh+':'+mm+':'+ss

    
s = input().strip()
result = timeConversion(s)
print(result)















