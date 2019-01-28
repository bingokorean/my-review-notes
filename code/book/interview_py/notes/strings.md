# Strings

String은 character들로 구성된 특별한 array라 볼 수 있다. 그러나 string와 array를 구분지으려 한다. comparison, joining, splitting, searching for substrings, replacing one string by another, parsing 등 string에만 사용되는 operation이 있기 때문이다.

memory에서 string이 어떻게 표현되는지 이해해야 한다. basic string operation을 이해해야 한다. Advanced string processing 알고리즘은 보통 hash table, dynamic programming을 사용한다. 여기서는 string의 basic technique만 알아본다.

### String boot camp

Palindromic string은 그냥 읽으나 reversed하게 읽으나 똑같은 string이다. Palindromic한 string인지 체크하는 프로그램을 작성해보자. Input의 reverse를 위해 새로운 string을 만들지 말고, input string을 forward하게 backward하게 traverse하면서 space를 save하자. even과 odd length string을 균등하게 처리하는점을 잘 확인하자. 

```
def is_palindromic(s):
    # Note that s[~i] for i in [0, len(s) - 1] is s[-(i + 1)]
    return all(s[i] == s[~i] for in range(len(s) // 2))
```

time complexity는 O(n)이고 space complexity는 O(1) 이다. (n은 string 길이)

### Top Tips for Strings

* Array와 비슷하게 string 문제는 O(n) space를 가지는 simple brute-force solution을 종종 가진다. 그러나, O(1) space complexity로 줄이는 subtler solution도 있다.
* string type이 immutable한 영향(implication)을 이해해야 한다. (immutable string을 concatenate하려면 new string을 할당해야 함). 파이썬의 list와 같이 immutable string을 대안 타입을 알아야 한다.
* front에서 mutable string을 업데이트하는 일은 slow하다. back에서부터 value를 write하는 것이 좋다.

### Know your string libraries

String의 key operation과 function은 다음과 같다. <br>
s[3], len(s), s + t, s[2:4], s in t, s.strip(), s.startswith(prefix), s.endswith(suffix), 'Euclid,Axiom 5,Parallel Lines'.split(','), 3 * '01', ','.join(('Gauss', 'Prince of Mathematiccians', '1777-1855')), s.tolower(), 'Name {name}, Rank {rank}'.format(name='Archimedes', rank=3)

String이 immutable하다는 것을 항상 기억하고 있어야 한다. s = s[1:] or s+= '123'과 같은 operation은 항상 새로운 character array를 만들고 나서 s를 할당하는 식이다. (변형은 없다 조금만 바뀌어도 항상 새로 만든다) 이러한 특징은 a single character를 n 번동안 string에 concatenate한다고 하면, O(n sqaure) time complexity를 가지게 된다 (파이썬의 몇몇의 알고리즘은 under-the-hood 트릭을 사용하여 이러한 할당을 피하고 O(n)의 time complexity를 가지기도 한다)




