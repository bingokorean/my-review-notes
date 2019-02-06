# Stacks and Queues

> Linear list, in which insertions, deletions, and accesses to values, occur almost always at the first or the last node are very frequently encountered, and we give them special names ... - "The Art of Computer Programming, Volum 1," D. E. Knuth, 1997

Stack은 삽입과 삭제에 대해 last-in, first-out 정책을 가지고, Queue는 first-in, first-out 정책을 가진다. Stack와 Queue 모두 복잡한 문제를 풀기 위한 솔루션인 buliding blocks를 한다. 또한, 그들은 stand-alone 문제를 만들 수 있다.

## Stacks

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


## Queues

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








