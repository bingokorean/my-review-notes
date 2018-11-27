####################################################
""" Sherlock and Anagrams """
####################################################
''' by darkshadows, Medium, 50, https://www.hackerrank.com/challenges/sherlock-and-anagrams/problem?h_l=interview&isFullScreen=false&playlist_slugs%5B%5D%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D%5B%5D=dictionaries-hashmaps
두 개의 string이 서로 anagrams라는 의미는 재배열하면 똑같은 string을 가질 수 있다는 의미이다.
string 하나를 입력받아 anagram을 가지는 substring 개수를 출력하시오.
(단, substring이 같다하더라도 position이 다르면 고유의 substring으로 인정한다)
### Sample Input
2
abba
kkkk
### Sample Output
4 -> [a,a], [ab,ba], [b,b], [abb,bba]이므로 4개이다.
10 -> [k,k]가 6개, [kk,kk]가 3개, [kkk,kkk]가 1개로 총 10개이다. 
'''
#!/bin/python
import math
import os
import random
import re
import sys

# Complete the sherlockAndAnagrams function below.
def sherlockAndAnagrams(s):
    list_s = list(s)
    cnt  = 0
    k = 1 # 길이
    while(k!=len(list_s)+1):
        i = 0
        while(i!=len(list_s)-k+1):
            p = i # 위치
            while(p!=len(list_s)-k+1):
                if p==i: # 첫 번째 - 기준.
                    sorted_substr = ''.join(sorted(list_s[p:p+k])) # sort()함수를 사용하면 해쉬맵을 사용할 필요가 없다.
                elif p > i:
                    if sorted_substr == ''.join(sorted(list_s[p:p+k])): # sort()함수를 사용하면 해쉬맵을 사용할 필요가 없다.
                        cnt += 1
                else:
                    pass
                #print(sorted_substr, ''.join(sorted(list_s[p:p+k])), cnt)        
                p += 1
            #print('i plus 1')
            i += 1
        k += 1
    return cnt
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(raw_input())
    for q_itr in xrange(q):
        s = raw_input()
        result = sherlockAndAnagrams(s)
        fptr.write(str(result) + '\n')
    fptr.close()



####################################################
""" Hash Tables: Ransom Note """
####################################################
''' by saikiran9194, Easy, 25

< Sample Input 0 >
6 4
give me one grand today night
give one grand today
< Sample Output 0 >
Yes

< Sample Input 1 >
6 5
two times three is not four
two times two is four
< Sample Output 1 >
No
< Explanation 1 >
'two' only occurs once in the magazine.
'''
#!/bin/python3
import math
import os
import random
import re
import sys
# Complete the checkMagazine function below.
def checkMagazine(magazine, note):

    my_dic = dict()
    
    for token in magazine:
        if my_dic.get(token) == None:
            my_dic[token] = 1
        else:
            my_dic[token] += 1
            
    #print(my_dic)
    for token in note:
        #print(my_dic.get(token))
        if my_dic.get(token) == 0:
            print('No')
            return 0
        
        if my_dic.get(token) == None:
            print('No')
            return 0
        else:
            my_dic[token] -= 1
    
    print('Yes')
    return 0
    
    
if __name__ == '__main__':
    mn = input().split()
    m = int(mn[0])
    n = int(mn[1])
    magazine = input().rstrip().split()
    note = input().rstrip().split()
    checkMagazine(magazine, note)





####################################################
""" Two Strings """
####################################################
''' by zxqfd555, Easy, 25
< Sample Input >
2
hello
world
hi
world
< Sample Output >
YES
NO
< Explanation >
hello와 world는 공통으로 'l' char를 가지고 있으므로 YES,
hi와 world는 공통으로 가지는 char가 없음.
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
        if char_dict.get(x) == 'stored':
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





