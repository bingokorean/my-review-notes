# Arrays

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

