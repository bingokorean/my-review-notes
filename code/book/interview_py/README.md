# Elements of Programming Interviews in Python

### Contents

_The Interview_
* Ready, Strategy, Conduct an Interview

_Data Structures and Algorithms_ 
* [Primitive Types](#Primitive-Types)
* [Arrays](#Arrays)
* [Strings](#Strings)
* [Linked Lists](#Linked-Lists)
* [Stacks and Queus](#Stacks-and-Queues)
* [Binary Trees](#Binary-Trees)
* [Heaps](#Heaps)
* Searching
* Hash Tables
* Sorting
* Binary Search Trees
* Recursion
* [Dynamic Programming](#Dynamic-Programming)
* Greedy Algorithms and Invariants
* Graphs
* Parallel Computing

_Domain Specific Problems_ 
* Design Problems
* Language Questions
* Object-Oriented Design
* Common Tools

_The Honors Class_
* Honors Class


<br>










## Primitive Types

> Representation is the essence of programming - "The Mythical Man Month," F.P.Brooks,1975

프로그램은 메모리에 있는 변수(variable)를 명령어(instruction)에 따라 업데이트한다. 변수는 타입(type)을 가진다. 타입은 데이터에 대한 분류이다. 데이터는 해당 타입이 가질 수 있는 범위의 값과 명령을 가진다. 타입은 언어(language)에 의해 제공되고 프로그래머에 의해 정의된다. 파이썬에서는 모든 것이 객체(object)이다. Boolean, integer, character 등도 객체이다.

### Primitive types boot camp

nonnegative integer에서 1인 bit 개수를 세는 프로그램을 primitive type을 활용해서 작성하자. 다음 프로그램은 least-significant bit에서부터 shifting과 masking을 하면서 one-at-a-time 마다 bit를 테스트한다. 이 프로그램은 the size of the integer word의 hard-coding을 방지하는 법을 보여주고 있다.

```
def count_bits(x):
    num_bits = 0
    while x:
        num_bits += x & 1
        x >>= 1
    return num_bits
```

O(1)의 computation per bit를 가지고, O(n)의 time complexity를 가진다 (n은 integer를 표현하기 위한 bit 개수). (참고. 4.1 문제-facing page, 솔루션이 위의 프로그램 성능을 향상시킴.)

### Top Tips for Primitive Types

* bitwise operators (특히, XOR) 에 익숙해져야 한다
* mask 사용법을 알아야 하고, machine independent하게 생성할 수 있어야 한다.
* fast ways to clear the loweest set bit 을 알아야 한다. (나아가 set the lowermost 0, get its index 등의 fast way도 알아야 한다)
* signedness에 대한 이해와, signedness가 shifting에 주는 영향(implication)을 알아야 한다.
* operation을 가속화하기 위해 cache를 고려하고, cache를 brute-force small input에 사용할 줄 알아야 한다.
* 교환(commutativity), 결합(associativity)이 parallel과 reorder operation에서 사용될 수 있음을 알아야 한다.

### Know your primitive types

파이썬은 numerics(e.g. integer), sequences(e.g. list), mappings(e.g. dict), classes, instances, exceptions 들과 같은  built-in type 을 가진다. 이러한 타입들의 모든 인스턴스는 객체(object)이다. 

파이썬3에서 integer는 메모리가 지원하는 한 무한하다. `sys.maxize`를 통해 word-size를 확인할 수 있다. 예를 들어, 64-bit machine에서 miximum value integer는 `2**63 - 1` 만큼의 크기를 word에 할당받을 수 있다. Float의 bound는 `sys.float_info`에서 확인할 수 있다.

* bit-wise operator에 익숙해야 한다 (e.g. 6&4, 1|2, 8>>1, -16>>2, 1<<10, ~0, 15^x (음수는 2의 보수로 표현된다. integer는 infinite precision을 가지므로 파이썬에서는 signed shift 개념이 없다.)
* numeric tpye을 위한 중요 함수는 다음과 같다 - abs(-34.5), math.ceil(2.17), math.floor(3.14), min(x,-4), max(3.14,y), pow(2.71,3.14), `2.71**3.14`, math.sqrt(225)
* integer와 string 사이, float과 string을 상호전환하는 방법을 알아야 한다 - str(42), int('42'), str(3.14), float('3.14')
* integer와 다르게 float은 infinite precision이 아니다. 따라서, float('inf'), float('-inf')로 infinity를 표현한다. 이 점이 integer와 다른 점이고 pseudo max-int, pseudo min-int를 만들 때 사용된다. 
* floating point value들을 비교할 때 math.isclose()를 사용하면 좋다. 즉, very large value들을 비교할 때, absolute와 relative difference들을 처리할 수 있다.
* random에 대한 주요 함수는 다음과 같다 - random.randrange(28), random.randint(8.16), random.random(), random.shuffle(A), random.choice(A)

## 4.1





<br>
















## Arrays

> The machine can alter the scanned symbol and its behavior is in part determined by that symbol, but the symbols on the tape elsewhere do not affect the behavior of the machine - "Intelligent Machinery," A.M.Tuning,1948

Array는...
* contiguous block of memory
* useful for representing sequences
* retrieving and updating A[i] takes O(1) time
* inserting and deleting an element at index i takes O(n-i) time (n: length of the array)

### Top Tips for Arrays
* array problem은 보통 O(n) space complexity의 brute-force 솔루션을 가지지만, 자기 자신의 array만 사용하여 O(1) space complexity를 가지는 영리한 솔루션도 있다.
* array의 앞단에서 entry를 넣는 일은 느리다 (array 특성상 뒤의 entry들을 모두 이동해야 함) 따라서, array에 entry를 넣을 때는 뒷단에서 넣을 수 있는지 항상 확인하자.
* entry를 삭제하기보다는 (위와 같은 이유로), overwriting하는 법을 더 고려하자
* integer array를 다룰 때, array의 뒷쪽에 있는 숫자들을 처리하는 것을 고려하자. 또는, array를 reverse하여 덜 중요한 digit을 첫 번째 entry로 만들어라.
* subarrays를 다룰 줄 아는 능력이 중요하다
* 과거정보를 읽을 때 off-by-1 에러를 내기 쉬우니 항상 주의하자
* 2D arrays를 사용할 때, parallel logic (for rows and columns)을 염두해두자
* 가끔씩 analytically하게 문제를 푸는 것보다 일일이 simulate하는 것이 쉬울 때도 있다 (어렵게 방정식을 통해 어떤 패턴을 이해하는 것보다 for문을 돌려서 직접 패턴을 들여다보는 게 더 쉬울 때가 많다)

### Array boot camp
Array에 대한 good insight를 얻을 수 있는 예제이다. 숫자들의 array를 입력으로 받고 짝수들이 먼저 등장하도록 정렬해라. 만약 O(n) space를 사용한다면 문제를 쉽게 풀 수 있지만, **O(1) space**만으로(without allocating additional storage) 문제를 풀어보자.

```python
def even_odd_sort(integers):
    next_even, next_odd = 0, len(integers)-1
    while next_even < next_odd:
        print(integers)
        if integers[next_even] % 2 == 0:
            next_even += 1
        else:
            integers[next_even], integers[next_odd] = integers[next_odd], integers[next_even]
            next_odd -= 1    
```

* array를 다룰 때, both ends들을 효과적으로 다루는 법을 알면 좋다
* 가상으로 array를 3개의 subarray들로 구성되었다고 판단: Even - Unclassified - Odd 순서로
* 초기엔 Even과 Odd는 비어있고, 모두 Unclassified에 속함
* Unclassified를 순회하면서 짝수검사의 결과로 통해 swap으로 Even과 Odd의 boundary로 보낸다
* 순회가 진행될수록 Even과 Odd는 팽창하고 Unclassified는 줄어든다
* 이 알고리즘은 O(n) time complexity와 O(1) space complexity를 가진다

다음 예제를 통해 위의 함수가 어떤 식으로 진행되는지 알아보자.
```
test = [3,5,4,8,9,7,6,10,1,2]

even_odd_sort(test)
>>>
[3, 5, 4, 8, 9, 7, 6, 10, 1, 2]
[2, 5, 4, 8, 9, 7, 6, 10, 1, 3]
[2, 5, 4, 8, 9, 7, 6, 10, 1, 3]
[2, 1, 4, 8, 9, 7, 6, 10, 5, 3]
[2, 10, 4, 8, 9, 7, 6, 1, 5, 3]
[2, 10, 4, 8, 9, 7, 6, 1, 5, 3]
[2, 10, 4, 8, 9, 7, 6, 1, 5, 3]
[2, 10, 4, 8, 9, 7, 6, 1, 5, 3]
[2, 10, 4, 8, 6, 7, 9, 1, 5, 3]


test
>>>
[2, 10, 4, 8, 6, 7, 9, 1, 5, 3]
```

### Know your array libraries
파이썬에서 Arrays는 list type이다 (tuple도 list이지만 immutable하다). list의 큰 특징은 dynamically-resized하다 (크기에 대한 제한이 없고, 어떤 곳에서든지 값이 추가되고 삭제가 가능하다)

* list 초기화 - e.g. [3,5,7,11], [1]+[0]*10, list(range(100)), list-comprehension
* basic operation - e.g. len(A), A.append(42), A.remove(2), A.insert(3, 28)
* 2D array 초기화 - e.g. [[1,2,4], [3,5,7,9], [13]]
* array에 어떤 value가 있는지 checking하는 것은 O(n) time complexity를 가진다 (n은 array 길이)
* deep copy와 shallow copy에 대한 이해 필요 - B=A 와 B=list(A) 의 차이. 전자는 주소를 공유함.
* Key list methods - min(A), max(A), bisect.bisect(A,6), bisect.bisect_left(A,6), bisect.bisect_right(A,6), A.reverse() (in-place), reversed(A) (returns an iterator), A.sort() (in-place), sorted(A) (returns a copy), del A[i], del A[i:j]
* Slicing에 대한 이해 - A = [1,6,3,4,5,2,7] 를 예시로 살펴보자.
   * A[2:4]=[3,4], A[2:]=[3,4,5,2,7], A[:4]=[1,6,3,4], A[:-1]=[1,6,3,4,5,2], A[-3:]=[5,2,7], A[-3:1]=[5,2], A[1:5:2]=[6,2], A[5:1:-2]=[2,4], A[::-1]=[7,2,5,4,3,6,1]
   * slicing으로 rotate를 흉내낼 수 있다 - A[k:] + A[:k] -> rotates A by k to the left
   * B = A[:] does shallow copy of A into B

파이썬은 list comprehension을 통해 list를 멋있게 만들 수 있다. list comprehension은 다음과 같이 구성된다.
  * an input sequence
  * an iterator over the input sequence
  * a logical condition over the iterator (this is optional)
  * an expression that yields the elements of the derived list

예를 들어, `[x**2 for x in range(6)] = [0,1,4,9,16,25]` 이고 `[x**2 for x in range(6) if x % 2 == 0] = [0,4,16]` 이다. <br>
비록 list comprehension이 map(), filter(), lambdas로 재구성될 수 있지만, list comprehension이 더 읽기 쉽다 (lambda가 안쓰이므로) <br>
list comprehension은 multiple levels of looping을 지원한다. `[(x,y) for x in A for y in B]` 처럼.. 하지만, two nested comprehension 까지만 보통 사용을 권장하고 그 이상은 그냥 for-loop 사용을 권장한다. <br>
set과 dictionary도 list comprehension을 지원한다.

## 5.1 


<br>










## Strings

> String pattern matching is an important problem that occurs in many areas of science and information processing. In computing, it occurs naturally as part of data processing, text editing, term rewriting, lexical analysis, and information retrieval. - "Algorithms For Finding Patterns in Strings," A.V.Aho,1990

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


<br>










## Linked Lists

> The S-expressions are formed according to the following recursive rules. 1. The atomic symbols p1, p2, etc, are S-expressions. 2. A null expression is also admitted. 3. If e is an S-expression so is (e). 4. If e1 and e2 are S-expressions so is (e1,e2). - "Recursive Functions Of Symbolic Expressions," J.McCarthy,1959 

List는 하나의 (중복이 있을 수 있는) ordered collection of values를 만든다. 구체적으로, a singly linked list는 하나의 sequence of nodes를 포함하는 데이터 구조이다 (각 노드는 an object와 next node를 가리키는 a reference를 가진다). 첫 번째 node는 head라 불리고, 마지막 node는 tail이라 불린다 (tail의 next field는 null이다). a doubly linked list는 node가 양방향으로 reference들을 가지고 있다 (이로써 null로 마킹하는 대신에 self-loop를 사용할 수 있다). 

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/linked_list_1.PNG" width="80%" height="80%"></p>

List는 linear order적으로 object를 포함할 수 있다는 점에서 array와 비슷하다. 큰 차이점은 list에서 element를 삽입(inserting)과 삭제(deleting)할 때 time complexity가 O(1)인 점이고, (반면) k-th element를 찾을(obtaining) 때는 time complexity가 O(n)이다. List가 더 복합한 데이터 구조의 building block이지만, tricky 문제에서 유용히 활용될 수 있다.

이 장에서 있는 모든 문제에서 각 node는 두 가지 entry를 가진다 - data field, and next field.

```
class ListNode:
    def __init__(self, data=0, next=None):
        self.data = data
        self.next = next
```

### Linked lists boot camp

두 가지 타입의 list-related 문제가 있다 - 당신 스스로 lit를 구현하는 것과 표준 list 라이브러리를 활용하는 것. 

A singly linked list의 basic list API를 구현해보자 - search, insert, delete

```
# Search for a key
def search_list(L, key):
    while L and L.data != key:
        L = L.next
    # if key was not present in the list, L will have become null
    return L

# Insert a new node after a specified node
def insert_after(node, new_node):
    new_node.next = node.next
    node.next = new_nodes
    
# Delete the node past this one. Assume node is not a tail
def delete_after(node):
    node.next = node.next.next
```

삽입과 삭제는 local operation이고 O(1) time complexity를 가진다. 검색은 전체 list를 순회(traversing)하는 것이 필요하므로 (만약 key가 마지막 node에 있거나 없는 경우) O(n)의 time complexity를 가진다. (n은 node의 개수이다)

### Top Tips for Linked Lists

* List 문제는 O(n) space를 사용하는 simple brute-force solution을 가지지만, existing list node를 사용해서 O(1)까지 줄일 수 있는 subtler solution을 가지기도 한다.
* 많은 List 문제는 개념적으로 simple하고, 알고리즘을 디자인하는 것보다는 cleanly coding what's specified에 가깝다.
* Dummy head (=sentinel)을 사용하는 것을 권장한다. 이를 통해 empty list를 확인하는 것을 피할 수 있고, code를 간략화해주고, bug를 줄이도록 해준다.
* Head와 tail을 위해 Update next를 하는 것(and previous for doulbe linked lsit)을 잊기 쉬우니 주의하라.
* Singly linked list를 사용하는 알고리즘은 two iterators(, one ahead of the other, or one advancing quicker than the other)를 사용하는 것이 좋다.

### Know your linked list libraries

파이썬 list type은 dynamically resized array로 구현되어 있다 (Array편을 참조). 이 장에서의 Linked list는 파이썬의 표준 type을 반영하지 않는다. 우리는 signly and doubly linked list type을 정의한다.





<br>




## Stacks and Queues

> Linear list, in which insertions, deletions, and accesses to values, occur almost always at the first or the last node are very frequently encountered, and we give them special names ... - "The Art of Computer Programming, Volum 1," D. E. Knuth, 1997

Stack은 삽입과 삭제에 대해 last-in, first-out 정책을 가지고, Queue는 first-in, first-out 정책을 가진다. Stack와 Queue 모두 복잡한 문제를 풀기 위한 솔루션인 buliding blocks를 한다. 또한, 그들은 stand-alone 문제를 만들 수 있다.

### Stacks

Stack은 두 가지 basic operation(push(추가), pop(삭제))을 가진다. Element는 last-in, first-out 순서로 추가되거나 삭제된다. Stack이 만약 비었을 경우 pop은 null을 리턴하거나 예외를 발생시킨다. 

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/stacks_and_queues_1.PNG" width="80%" height="80%"></p>

Stack이 linked list로 구현될 경우, array로 구현될 경우,  array가 dynamically하게 resized되는 경우에도 pop, push 명령어는 O(1) time complexity를 가진다. Stack은 peek(returns the top of the stack without popping it)와 같은 추가적인 명령어도 가진다.

### Stacks boot camp

Stack의 LIFO 정책은 주어진 element로부터 step back을 하기 매우 어렵거나 불가능한 시퀀스의 reversse iterator를 만들기에 매우 유용하다. 다음 프로그램은 stack을 이용해 singly-linked list를 reverse order로 프린트한다. Time & space complexity가 O(n)이다 (n은 list에 있는 node의 개수). 

```
def print_linked_list_in_reverse(head):
    nodes = []
    while head:
        nodes.append(head.data)
        head = head.next
    while nodes:
        print(nodes.pop())
```

또 다른 방법으로, 7.2 solution 방법을 사용하면 O(n) time complexity와 O(1) space complexity를 가진다. 

### Top Tips for Stacks

* 언제 stack의 LIFO 정책이 적용되는지 이해해야 한다. 예를 들어, parsing은 일반적으로 stack의 혜택을 받는다.
* finding the maximum element와 같은 추가적인 명령어를 지원하기 위해 basic stack 또는 queue 자료 구조를 augmenting하는 것을 고려하라.

### Know your stack libraries

your own stack class를 구현하거나, built-in list-type을 사용하면 된다.

* s.append(e) : pushes an element onto the stack. (not much can go wrong with a call to push)
* s[-1] : retrieve, but does not remove, the element at the top of the stack
* s.pop() : remove and return the element at the top of the stack
* len(s) == 0 : tests if the stack is empty
* s가 empty list인 경우, s[-1]과 s.pop()은 IndexError exception을 발생시킨다.

## 8.1

...


### Queues

Queue는 enqueue와 dequeue라는 두 가지 basic operation을 가진다 (queue가 empty면 dequeue는 일반적으로 null을 리턴하거나 exception을 발생시킨다). First-in, first-out 순서로 element가 추가(enqueued)되고 삭제(dequeued)된다. 가장 최근에 삽입된 element는 tail 또는 back element라 부르고, 가장 오래 전에 삽입된 element는 head 또는 front element라 부른다. 

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/stacks_and_queues_2.PNG" width="80%" height="80%"></p>

Queue는 linked list로 구현하면 enqueue와 dequeue가 모두 O(1) time complexity를 가진다. Queue API는 다른 명령어도 가진다 - e.g. a method that returns the item at the head of the queue without removing it, a method that returns the item at the tail of the queue without removing it, etc. Queue는 array로도 구현될 수 있다 (8.7 참고)

Deque (doubly-ended queue)라는 확장된 버전의 queue가 있다. A doubly linked list로 구성되고, 모든 insertion과 deletion이 one of the two ends of the list에서 발생한다 (즉, head 또는 tail에서). Front에서의 insertion은 push라 부르고, back에서의 insertion은 inject이라 부른다. Front에서의 deletion은 pop이라 부르고, back에서의 deletion은 eject라 부른다 (언어와 라이브러리마다 명명법이 다를 수 있다)

### Queues boot camp

Basic queue API를 구현해보자 - enqueue, dequeue, max-method (returns the maximum element stored in the queue). Basic idea는 composition을 사용하는 것이다: a library queue object를 참조하는 private field를 추가하고, existing method (enqueue, dequeue, ..)를 해당 object에 전달한다. 

```
class Queue:
    def __init__(self):
        self._data = collections.deque()      # deque지만 여기선 queue기능까지만 활용한다
    
    def enqueue(self, x):
        self._data.append(x)
        
    def dequeue(self):
        self._data.popleft()
        
    def max(self):
        return max(self._data)
```

Library queue와 같이 enqueue와 dequeue의 time complexity는 O(1)이다. 단, finding miximum의 time complexity는 O(n)이다 (n은 entry 개수). 커스텀 버전의 8.9 솔루션으로 finding maximum의 time complexity가 O(1)으로 만들 수 있다.

### Top Tips for Queues

* queue FIFO 특징을 이해하고 언제 적용할 수 있는지 알아야 한다. 예를 들어, queue는 순서를 유지하고 싶을 때 사용하기 이상적이다.

### Know your queue libraries

your own queue class를 구현할 수 있고, collection.deque class를 사용할 수 있다.

* q.append(e) : push an element onto the queue (not much can go wrong with a call to push)
* q[0] : retrieve, but not remove, the element at the front of the queue
* q[-1] : retrieve, but not remove, the element at the back of the queue
* q.popleft() : remove and return the element at the front of the queue
* Dequeing or accessing the head/tail of an empty collection results in an IndexError exception being raised


## 8.6

...



<br>











## Binary Trees

> The method of solution involves the development of a theory of finite automata operating on infinite trees. - "Decidability of Second Order Theories and Automata on Trees," M.O.Rabin,1969

Formally, a binary tree는 empty 이거나 a left binary tree와 a right binary tree를 가지는 a root node r 이다. Subtree 자체가 binary tree이다. Left binary tree를 left subtree of the root라 부르고, right binary tree는 right subtree of the root라 부를 수 있다. 

Binary tree는 대부분, 정렬된 key가 저장되는 binary search tree에서 사용된다. 뿐만 아니라 binary tree를 사용하는 많은 application이 있다. High-level 관점에서 보면, binary tree는 hierarchy를 다룰 때 적용하기 적합하다.

다음은 binary tree의 graphical representation이다. Node A는 root이다. Node B와 I는 A의 left와 right children이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/binary_trees_1.PNG" width="90%" height="90%"></p>

Node는 additional data를 저장한다. Node의 prototype은 다음과 같다.

```python
class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
```

Root를 제외한 모든 node는 left subtree의 root이거나 right subtree의 root가 될 수 있다. l이 p의 left subtree의 root라고 하면, l은 p의 left child라 할 수 있고, p는 l의 parent라 할 수 있다 (right child도 같은 이야기). 한 node가 p의 left child 또는 right child일 경우, 해당 node를 p의 child라 부른다. 주목할 점은 root를 제외하고, 모든 node는 unique parent를 가진다. 항상 그렇지는 않지만, node object를 정의할 때 parent field를 포함한다 (단, root node는 null로). 각 node는 root에서부터 자기자신의 node까지의 unique sequence of nodes 정보를 가질 수 있다. 이러한 sequence는 search path라 부르기도 한다. 

Parent-child 관계는 binary tree에서 ancestor-descendant 관계로 정의한다. 구체적으로, 어떤 node가 d node의 anscestor라고 한다면, 그 node는 root에서부터 d까지의 search path에 포함된다고 할 수 있고, d node는 그 node의 descendant라 할 수 있다. (저자의 convention으로) Node 자기 자신은 ancestor 그리고 descendant가 될 수 있다. 자기 자신을 제외하고 descendant가 없는 node를 leaf라고 부른다.

Node n의 depth는 root에서부터 n까지의 (n 자기자신은 제외) search path에 포함된 node 개수를 말한다. Binary tree의 height는 tree에 있는 node들 중에서 가장 큰 depth를 말한다. Tree의 level은 모든 node가 똑같은 depth를 가지는 경우에만 정의할 수 있다. (위 그림 참고)

(Figure 9.1 참고) Node I는 J와 O의 parent node이다. Node G는 B의 descendant node이다. Node L까지의 search path는 <A,I,J,K,L>이다. Node N의 depth는 4이다. Node M은 maximum depth 5를 가지고, 이에 따라 tree의 height는 5이다. (주의) Node B를 root(기준)로 하고 이의 subtree의 height는 3이다. 반면, the height of the subtree rooted at H는 0이다. Node D, E, H, M, N, P는 tree의 leaf들이다.

Full binary tree란 leaf node를 제외한 모든 node가 2개의 children을 가지고 있는 tree를 말한다. Perfect binary tree는 full binary tree임과 동시에 모든 leaf node들이 똑같은 depth를 가진 tree를 말한다. Complete binary tree는 (아마도) 마지막을 제외한 모든 레벨이 완벽히 채워져 있고, 모든 노드들이 최대한 왼쪽인 상태이다. Full binary tree의 nonleaf node의 개수는 leaf node의 개수보다 적다. Perfect binary tree의 height h는 정확하게 2^(h+1)-1 nodes를 가지고 2^h는 leaf들이다. n node를 가지는 complete binary tree의 height는 [logn]이다. Left-skewed tree는 right child가 단 한개도 없는 node들로 구성된다. Right-skewed tree는 반대로 left child가 단 한개도 없는 node들로 구성된다. 위의 두 가지 중 하나라면, 우리는 binary tree를 skewed 되었다고 표현한다.

Binary tree의 key computation은 tree의 모든 노드들을 **traversing**하는 것이다. (Traversing 또는 walking이라고도 불린다) 

* [Inorder traversal] traverse the left subtree, visit the root, then traverse the right subtree. <D,C,E,B,F,H,G,Z,J,L,M,K,N,I,O,P>
* [Preorder traversal] visit the root, traverse the left subtree, then traverse the right subtree. <A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P>
* [Postorder traversal] traverse the left subtree, traverse the right subtree, and then visit the root. <D,E,C,H,G,F,B,M,L,N,K,J,P,O,I,A>

n nodes와 height h를 가지는 T라는 binary tree가 있다고 하자. Recursively하게 구현하면, traversal은 O(n) time complexity와 O(h) addtional space complexity를 가진다. (space complexity는 maximum depth of the function call stack이다). 각 노드가 parent field를 가질 때 마다 traversal은 추가적인 O(1) space complexity를 가진다. (tree 용어는 overloaded되었으므로 혼란을 불러일으킬 수 있다)

### Top Tips for Binary Trees

* Recursive 알고리즘은 tree 문제에 잘 어울린다. Function call stack을 할당할 때마다 space를 추가하는 것을 기억하자.
* 어떤 tree 문제는 O(n)을 가지는 간단한 brute-force 솔루션을 가지지만, existing tree node를 사용하여 O(1)까지 줄일 수 있는 영리한 솔루션도 가진다.
* Complexity 분석을 할 때, left- and right-skewed tree를 고려하자. h height인 tree의 O(h) complexity를 balanced tree로 만들면 O(logn)으로 변환할 수 있다. 하지만, skewed tree는 O(n) complexity를 가진다.
* 각 node가 parent field를 가지면, code를 더 심플하게 만들 수 있고, time과 space complexity를 줄일 수 있다.
* 실수로 single child를 leaf로 판단할 수 있으니 조심하자.

### Binary trees boot camp

binary tree에 speed를 높이기 위한 좋은 방법은 다음과 같은 세 가지의 기본적인 traversal (inorder, preorder, postorder)을 구현하는 것이다.

```python
def tree_traversal(root):
    if root: # 이게 recursive 함수의 탈출 조건이 되기도 한다.
	    # Preorder: Processes the root before the traversals of left and right
		# children.
		print('Preorder: %d' % root.data)
		tree_traversal(root.left)
		# Inorder: Processes the root after the traversal of left child and
		# before the traversal of right child
		print('Inorder: %d' % root.data)
		tree_traversal(root.right)
		# Postorder: Processes the root after the traversals of left and right
		# children
		print('Postorder: %d' % root.data)
```

각 approach의 time complexity는 O(n)이다 (n는 tree에서 node의 개수임). Function call stack은 최대 tree의 depth h를 가지므로 space complexity는 O(h)이다. Height h의 최소값은 log(n)이고 (complete binary tree), h의 최대값은 n이다 (skewed tree). 




<br>












## Heaps

Heap은 특별한 binary tree이다. 구체적으로, 특별한 complete binary tree와 같다. Heap이 되려면, 다음과 같은 heap property들을 만족해야 한다. 각 노드에서의 키(key) 값은 그의 자식의 키 값보다 항상 크거나/작아야 한다. 다음 그림의 (a)는 max-heap의 예시를 보여주고 있다. Max-heap은 array로 구현될 수 있다. 인덱스 i에 있는 노드의 자식은 인덱스 2i+1 과 2i+2가 된다. 그림 (a)의 array representation은 [561, 314, 401, 28, 156, 359, 271, 11, 3] 이 된다. 

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/heaps_1.PNG" width="90%" height="90%"></p>

Max-heap은 O(log(n))의 insertion을 제공하고, max element에 대해 O(1) time lookup을 제공하고, max element를 delete하는 것에 대해 O(log(n))을 제공한다. Extract-max 명령어는 delete와 maximum element를 return하는 것으로 정의된다. 그림 (b)는 max element의 deletion을 나타낸다. Arbitrary element를 searching하는 것은 O(n) time complexity가 소요된다. 

Heap은 가끔 priority queue라고도 불리운다. 이유는 queue와 행동이 비슷하기 때문이다. 단 하나의 차이점으로, 각 element는 그와 연관된 'priority'를 가지고, deletion은 highest priority를 가지는 element를 삭제한다.

Min-heap은 max-heap과 비교해 완전히 symmetric하고, minimum element에 대해 O(1) time lookup을 제공한다. 


## Heaps boot camp

스트리밍 모드로 string들로 구성된 시퀀스를 받는 프로그램을 작성한다고 하자. 스트리밍이기에 이전의 string을 보는 것과 같은 백업을 하지 못한다. 프로그램은 시퀀스에서 k longest string들을 찾아야 한다 (오직 k longest string만 찾으면 될 뿐 정렬할 필요는 없다). 

Min-heap (not max-heap!)이 이 문제에 올바른 자료구조이다. Min-heap를 통해 find-min, remove-min, 그리고 insert를 효율적으로 할 수 있다. 다음은 custom compare function (길이에 맞게 정렬)을 heap과 함께 사용하여 위의 문제를 푼 코드이다.

```python
def top_k(k, stream):
    # Entries are compared by their lengths
    min_heap = [(len(s), s) for s in itertools.islice(stream, k)] # -> [(5, 'hello'), (2, 'hi')]
    heapq.heapify(min_heap) # -> [(2, 'hi'), (5, 'hello')]
    for next_string in stream:
        # push next_string and pop the shortest string in min_heap
        heapq.heappushpop(min_heap, (len(next_string), next_string))
    return [p[1] for p in heapq.nsmallest(k, min_heap)]

top_k(2, ["hello", "hi", "goodbye", "oh...!", "thisiskorea!"])
>>>
['goodbye', 'thisiskorea!']
```

각각의 string은 O(log(k)) time으로 처리된다. 이 시간은 add, remove the minimum element from the heap을 말한다. 따라서, 입력에 n개의 string이 있다면, time complexity는 O(nlog(k))가 된다. 

Best-case time complexity를 가질 수 있다: 새로운 string 길이를 top of the heap의 string 길이와 비교를 먼저 한다 (이는 O(1) time). 그리고 새로운 string 길이가 too short to be in the set 하면... insert하는 것을 skip한다.


## Top Tips for Heaps

* 다음 같은 경우 heap을 사용하는 것을 고려해라: largest 또는 smallest elements에 대한 모든 일들을 처리할 때와 무작위의 element에 대한 lookup, delete, 또는 search 명령에 대해 빠른 처리가 필요하지 않을 경우.
* Heap은 다음과 같은 경우에 좋은 선택이 될 것이다: 하나의 collection 속에서 k largest 또는 k smallest elements를 계산할 때. 전자는 min-heap을, 후자는 max-heap을 사용한다.

## Know your heap libraries

파이썬에서 heap functionality는 `heapq` 모듈에서 제공된다. 

* heapq.heapify(L): transforms the elements in L into a heap in-place
* heapq.nlargest(k, L): returns the k largest elements in L
* heapq.nsmallest(k, L): returns the k smallest elements in L
* heapq.heappush(h, e): pushes a new element on the heap
* heapq.heappop(h): pops the smallest element from the heap
* heapq.heappushpop(h, a): pushs 'a' on the heap and then pops and returns the smallest element
* e = h[0]: returns the smallest element on the heap without popping it

`heapq`는 오로지 min-heap functionality만 제공하는 것을 기억하자. (integer 또는 float을 사용하는) Max-heap을 만들고 싶으면, 그들의 negative값을 insert하라. Object를 위해서는 `__lt()__`를 적절히 구현해라. Problem 10.4에 max-heap을 어떻게 사용하는지 나타내고 있다.




<br>



## Dynamic Programming

다이나믹 프로그래밍(DP)은 부분-문제(subproblems)로 쪼개질 수 있는 최적화, 검색, 카운팅 문제를 풀기 위한 일반적인 솔루션이다. 만약, 찾고자 하는 솔루션이 부분-문제(subproblems)와 연관되어 있다면, DP를 사용하는 것을 고려해야 한다. 

분할 정복 알고리즘과 비슷하게도, DP는 여러 개의 작은 문제들의 솔루션을 결합하면서 문제를 푼다. 다른 점은 똑같은 부분-문제(subproblems)가 재등장하는 것이다. 그러므로 효율적인 DP를 만드는 핵심은 중간 연산 결과물을 캐싱하는 데 있다. 

DP의 내면에 숨겨진 아이디어를 묘사하기 위해서 Fibonacci numbers를 계산하는 문제를 보자. 첫 두 개의 피보나치 숫자는 0과 1이다. 연속된 숫자들은 두 개의 이전 숫자들의 합으로 표현된다. 예를 들어, 0, 1, 1, 2, 3, 5, 8, 13, 21, ... 와 같이 나열된다. 피보나치 숫자는 다양한 어플리케이션에서 활용된다 - 바이오, 데이터 구조 분석, 병렬 컴퓨팅 등

수학적으로 n번째의 피보나치 숫자 F(n)은 F(n) = F(n-1) + F(n-2)로 표기되며, 여기서 F(0)=0 그리고 F(1)=1이다. F(n)을 재귀적으로 계산하는 함수는 n의 길이가 늘어날수록 기하급수적(exponential)으로 시간 복잡도(time complexity)가 증가한다. 왜냐하면 재귀적 함수는 F(i)들을 반복적으로 계산하기 때문이다. 다음 그림을 보면 F(i) 함수가 똑같은 인자를 가지고 반복적으로 콜하는 것을 확인할 수 있다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/dp_tree.PNG" width="100%" height="100%"></p>

중간 결과물을 캐싱하는 것은 피보나치 숫자의 시간 복잡도를 n 기준 선형으로 만들 수 있다. 여기서 저장 비용은 O(n)을 가진다.

```python
def fibonacci(n, cache={}):
	if n <= 1:
		return n
	elif n not in cache:
		cache[n] = fibonacci(n-1) + fibonacci(n-2)
	return cache[n]
```

캐싱 스페이스를 줄이는 것은 DP에서 반복해서 다루는 주제이다. 이제 F(n) 함수를 O(n) 시간과 O(1) 공간 복잡도를 가지는 프로그램을 알아보자. 위의 프로그램과 비교해서 아래의 프로그램은 캐쉬의 공간 복잡도를 줄이기 위해서 "bottom-up" 방식으로 캐쉬를 반복해서 채운다. 이는 캐쉬를 재사용 가능하게 해준다.

아하! 위의 프로그램은 재귀형이고 아래는 반복형이다. 근데 반복형이 공간 복잡도에서 더 이득을 가지네.? 재귀형은 "top-down" 방식이고 반복형은 "bottom-up" 방식이다.

```python
def fibonacci(n):
	if n <= 1:
		return n
	f_minus_2, f_minus_1 = 0, 1
	for _ in range(1, n):
		f = f_minus_2 + f_minus_1
		f_minus_2, f_minus_1 = f_minus_1, f
	return f_minus_1
```

DP 문제의 솔루션을 찾는 핵심은 다음과 같은 조건에서 문제를 부분-문제(subproblems)로 쪼갤 수 있는 방법을 찾는 것이다. 

* 문제를 부분-문제로 풀 수 있으면 상대적으로 문제를 풀기가 쉬워진다. 
* 이러한 부분-문제들은 캐시된다.

보통 (항상 그렇지는 않지만) 부분-문제는 찾기는 어렵지 않다.

여기에 조금 더 복잡한 DP 문제가 있다. 주어진 integer의 array에서 sub-arrays의 최대 합을 구하는 문제이다. 다음 그림에서 최대 subarray를 찾으면 인덱스 0에서 인덱스 3이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/dp_array.PNG" width="80%" height="80%"></p>

모든 경우의 수의 subarray의 sum을 계산하는 brute-force 알고리즘은 O(n^3) 시간 복잡도를 가진다. subarray의 개수는 n(n+1)/2 가 있고, 각 subarray를 sum을 계산하는 것은 O(n) time이 걸린다.

이 brute-force 알고리즘은 O(n^2)으로 성능에 O(n)의 추가적인 저장 공간으로 성능이 향상될 수 있다. S[k] = 모든 k에 대한 A[0,k] 합. 을 미리 계산하고 저장함으로써...

A[i,j]의 합은 S[j] - S[j-1]로 계산된다. 여기서 S[-1]은 0이 된다. 

이것은 사실 divide-and-conquer 알고리즘이다. A의 정중앙 인덱스 m을 n/2로 얻는다. subarray L=A[0,m] 과 R=A[m+1,n-1] 을 계산하면서 문제를 해결한다. 각각에 대해 문제를 푸는 것뿐만 아니라 subarray sum의 최대 l을 구하고 (여기서 subarray의 마지막 엔트리는 항상 L의 마지막 엔트리이다), subarray sum의 최대 r을 구한다 (여기서 subarray의 첫 번째 엔트리는 항상 R의 첫 번째 엔트리이다).

A에 대한 최대 subarray sum







## Dynamic Programming boot camp

위에서 언급한 Fibonacci numbers와 maximum subarray sum이 DP의 좋은 예시가 된다.

## Top Tips for Dynamic Programming






