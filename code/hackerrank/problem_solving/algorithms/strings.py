
####################################################
""" Strings: Making Anagrams """
####################################################
''' by amititkgp, Easy, 25, https://www.hackerrank.com/challenges/ctci-making-anagrams/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=strings
Given two strings, a and b, that may or may not be of the same length, 
determine the minimum number of character deletions required to make a and b anagrams.

### Sample Input
cde
abc
### Sample Output
4
< Explanation >
We delete the following characters from our two strings to turn them into anagrams of each other:
    Remove d and e from cde to get c.
    Remove a and b from abc to get c.
We must delete 4 characters to make both strings anagrams, so we print 4 on a new line.
'''
#!/bin/python3
import math
import os
import random
import re
import sys

# Complete the makeAnagram function below.
def makeAnagram(a, b):
    a_list = list(a)
    b_list = list(b)
    
    #delete_a = [x for x in a_list if not x in b_list]
    #delete_b = [y for y in b_list if not y in a_list]
    
    remain_a = [x for x in a_list if x in b_list]
    remain_b = [y for y in b_list if y in a_list]
    
    duplicate_a = remain_a[:]
    duplicate_b = remain_b[:]
    
    for aa in remain_a:
        try:
            duplicate_b.remove(aa)
        except ValueError:
            pass
    
    for bb in remain_b:
        try:
            duplicate_a.remove(bb)
        except ValueError:
            pass

    #print(len(a_list), len(b_list))
    
    #print(len(delete_a), len(delete_b))
    #print(delete_a)
    #print(delete_b)
    
    #print(len(remain_a), len(remain_b))
    #print(remain_a)
    #print(remain_b)
    
    
    return len(a_list) - len(remain_a) + len(b_list) - len(remain_b) + len(duplicate_a) + len(duplicate_b)
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    a = input()
    b = input()
    res = makeAnagram(a, b)
    fptr.write(str(res) + '\n')
    fptr.close()


    
######################################################################
""" Palindrome Index """
######################################################################
''' (Easy, 25, by amititkgp), https://www.hackerrank.com/challenges/palindrome-index/problem
Given a string of lowercase letters in the range ascii[a-z], determine a character that can be removed to make the string a palindrome 
문자를 적절히 삭제하여 앞에서부터 읽으나 뒤에서부터 읽으나 같은 어구(=palindrome index)를 만들어라
주어진 string에서 palindrome이 아닐 경우 무조건 한 개의 char만 삭제하면 palindrome이 된다는 가정을 한다
### Sample Input
3
aaab
baa
aaa
### Sample Output
3   -> index 3에 있는 b를 삭제
0   -> index 0에 있는 b를 삭제
-1   -> 이미 palindrome이므로 -1을 출력
'''

import sys
# (첫 번째 시도: got 10.72 score (total: 25.0) because of timeout. 시간초과로 5~7문제정도 풀지 못하였음)
'''
def is_palindrome(listed_s, len_s, cen_idx): # 데칼콜마니
    
    if len_s % 2 == 0: odd_even = 'even'
    else: odd_even = 'odd'    

    if odd_even == 'even':
        left = listed_s[:cen_idx+1] 
        right = listed_s[cen_idx+1:]
        right.reverse() # just execute
        if left == right:
            return True
        else:
            return False
        
    elif odd_even == 'odd':
        left = listed_s[:cen_idx+1] 
        right = listed_s[cen_idx+2:] # +1 because of odd
        right.reverse() # just execute
        if left == right:
            return True
        else:
            return False
        
def palindromeIndex(s):

    # Initialization   
    listed_s = list(s)
    len_s = len(listed_s)
    cen_idx = int(len_s/2)-1  # if [1,2,3,4] or [1,2,3,4,5], center is [2]

    # Is palindrome?
    if is_palindrome(listed_s, len_s, cen_idx) == True:
        return -1
    else: # if is not palindrome,
        # Search
        for i in range(0, len_s):
            temp_listed_s = listed_s[:]
            temp_listed_s.pop(i) # just execute
            len_s = len(temp_listed_s)
            cen_idx = int(len_s/2)-1
            if is_palindrome(temp_listed_s, len_s, cen_idx) == True:
                return i
            
        # ok, palindrome not exist.
        return -1
        
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = palindromeIndex(s)
    print(result)
'''
  
# implemented by 'straemer'    
# process with string not convert it into list.
t = int(input())
for testCase in range(t):
    testString = input()
    reverseString = testString[::-1]
    if testString == reverseString:
        print(-1) # 이미 palindrome이다
        continue
    for i in range(len(testString)):
        if testString[i] != reverseString[i]: # 이 부분이 나와의 큰 차이점. 전체를 비교하는 것이 아닌 하나의 char만 비교하면서 속도를 줄일 수 있음.
            removed = testString[:i] + testString[i+1:] # 
            if removed == removed[::-1]: # 삭제 후 palindrome인가? (입력이 baa인 경우)
                print(i) 
            else: # 입력이 aaab인 경우
                print(len(testString)-i-1) # 현재 index가 아닌 그 반대편에 있는 index를 사실 삭제했어야 했음.
            break    

    
    
######################################################################
""" Two String """
######################################################################     
''' by zxqfd555, Easy, 25, https://www.hackerrank.com/challenges/two-strings/problem
두 개의 string을 입력으로 받아, common substring을 가지는지 확인해보자 (substring은 one character도 가능)
### Sample Input
2
hello
world
hi
world
### Sample Output
YES
NO   -> hi와 world는 한 개의 character라도 공유하지 않는다
'''
#!/bin/python3
import math
import os
import random
import re
import sys

# Complete the twoStrings function below.
def twoStrings(s1, s2):
    char_dict = dict()
    for x in s1:
        char_dict[x] = 'stored'
    for x in s2:
        if char_dict.get(x) == 'stored': # 한 개의 character라도 공유하면 'YES'
            return 'YES'
    return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    for q_itr in range(q):
        s1 = input()
        s2 = input()
        result = twoStrings(s1, s2)
        fptr.write(result + '\n')
    fptr.close()

    
    
######################################################################
""" Super Reduced String """
######################################################################      
''' by amitikgp, Easy, 20, https://www.hackerrank.com/challenges/alternating-characters/problem?h_l=interview&isFullScreen=false&playlist_slugs%5B%5D%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D%5B%5D=strings
A와 B로만 구성된 string을 입력받고, 같은 char가 연이어 나타나지 않도록 char 제거 횟수를 출력하시오.
예) s = AABAAB, 0번째와 3번째 A를 제거한다. 따라서, 2개의 제거횟수를 가진다.
### Sample Input
5
AAAA
BBBBB
ABABABAB
BABABA
AAABBB
### Sample Output
3
4
0
0
4
'''
#!/bin/python
import math
import os
import random
import re
import sys

# Complete the alternatingCharacters function below.
def alternatingCharacters(s):
    list_s = list(s)
    cnt = 0
    cur_ch = ''
    
    for i, _ in enumerate(list_s):
        if i==0:
            cur_ch = list_s[i]
        else:
            if cur_ch == list_s[i]:
                cnt += 1
            else:
                cur_ch = list_s[i]
    return cnt

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(raw_input())
    for q_itr in xrange(q):
        s = raw_input()
        result = alternatingCharacters(s)
        fptr.write(str(result) + '\n')
    fptr.close()



######################################################################
""" Super Reduced String """
######################################################################            
''' by harshil7924, Easy, 10, https://www.hackerrank.com/challenges/reduced-string/problem
근접한 pair 문자들을 삭제해보자 (pair는 짝수 개수의 문자 집합이라 생각해도 됨
### Sample Input
aaabccddd
### Sample Output
abd
< Explanation 0 >
Steve performs the following sequence of operations to get the final string:
aaabccddd → abccddd → abddd → abd
'''
#!/bin/python3

import math
import os
import random
import re
import sys
 
# Complete the superReducedString function below.
def superReducedString(s):
    
    str_li = list(s)
    while True: # <- 주의: 겉을 while문으로 wrapping하지 않아서 처음엔 실패함.
    
        for i, each in enumerate(str_li):
            if i==0: # 바로 직전의 데이터를 확인하는 로직이므로 첫 번째 index는 패스한다.
                continue
            if str_li[i-1] == str_li[i]:
                # 한 꺼번에 2개의 쌍을 1로 변경하는 것이 중요하다
                str_li[i-1] = '1' 
                str_li[i] = '1'
        
        if len(str_li)==0:
            return 'Empty String'
        if not '1' in str_li:
            return ''.join(str_li)
        
        str_li = [x for x in str_li if x!='1'] # 1을 제외한 string을 새로 만들어 다시 시작할 준비를 한다

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = superReducedString(s)
    fptr.write(result + '\n')
    fptr.close()




