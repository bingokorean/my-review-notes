# Linked Lists

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


