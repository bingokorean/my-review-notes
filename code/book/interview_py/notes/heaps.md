# Heaps

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

`heapq`는 오로지 min-heap functionality만 제공하는 것을 기억하자. (integer 또는 float을 사용하는) Max-heap을 만들고 싶으면, 그들의 negative값을 insert하라. Object를 위해서는 __lt()__를 적절히 구현해라. Problem 10.4에 max-heap을 어떻게 사용하는지 나타내고 있다.
