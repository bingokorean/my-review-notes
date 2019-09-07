# 6. 내장 모듈

몇몇 표준 내장 패키지는 언어 사양의 일부이므로 파이썬의 특징과 밀접하게 관련있다. 이런 기본적인 내장 모듈은 복잡한 트로그램을 작성하거나 오류가 발생할 가능성이 큰 프로그램을 작성할 때 특히 중요하다.

## 42.

## 43.

## 44. 

## 45. 

## 46. 내장 알고리즘과 자료 구조를 사용하자

많은 데이터를 처리하는 파이썬 프로그램을 구현하다 보면 (파이썬 언어의 속도 때문이 아닌) 여러분이 작성한 코드의 알고리즘 복잡도 때문에 속도가 떨어지는 현상을 보게 된다. 최적의 알고리즘과 자료 구조를 사용할 필요가 있다.

파이썬 표준 라이브러리에는 필요한 만큼 많은 알고리즘과 자료 구조가 있다. 

#### Double-ended Queue

collection 모듈의 deque 클래스는 더블 엔디드 큐이다. Deque는 큐의 처음과 끝에서 아이템을 삽입하거나 삭제할 때 항상 일정한 시간이 걸리는 연산을 제공한다. 이는 선입선출(FIFO, Fist-In-First-Out) 큐를 만들 때 이상적이다.

```python
fifo = deque()
fifo.append(1)        # 생산자
x = fifo.popleft()    # 소비자
```

내장 타입 리스트(list)도 큐와 같이 순서가 있는 아이템 시퀀스를 담는다. 일정한 시간 내에 리스트의 끝에서 아이템을 삽입하거나 삭제할 수 있다. 하지만, 리스트의 시작 부분에서 아이템을 삽입하거나 삭제하는 연산에는 선형적 시간(linear time)(*횟수에 따라 시간도 늘어남)이 걸리므로 deque의 일정한 시간보다 훨씬 느리다.

#### 정렬된 Dictionary

표준 딕셔너리는 정렬되어 있지 않다. 즉, 같은 키와 값을 담은 딕셔너리를 순회해도 다른 순서가 나올 수 있다. 이런 동작은 딕셔너리의 빠른 해시 테이블을 구현하는 방식이 만들어낸 뜻밖의 부작용이다.

```python
a = {}
a['foo'] = 1
a['bar'] = 2

# 무작위로 'b'에 데이터를 추가해서 해시 충돌을 일으킴
while True:
    z = randint(99, 1013)
    b = {}
    for i in range(z):
        b[i] = i
    b['foo'] = 1
    b['bar'] = 2
    for i in range(z):
        del b[i]
    if str(b) != str(a):  # 여기서 단지 순서가 다르다고 break된다.
        break

print(a)
print(b)
print('Equal?', a==b)

>>>
{'foo': 1, 'bar': 2}
{'bar': 2, 'foo': 1}
Equal? True
```

collection 모듈의 OrderedDict 클래스는 키가 삽입된 순서를 유지하는 특별한 딕셔너리 타입이다. OrderedDict의 키를 순회하는 것은 예상 가능한 동작이다. 따라서 모든 코드를 확정하고 만들 수 있으므로 테스팅과 디버깅을 아주 간단하게 할 수 있다.

```python
a = OrderedDict()
a['foo'] = 1
a['bar'] = 2

b = OrderedDict()
b['foo'] = 'red'
b['bar'] = 'blue'

for value1, value2 in zip(a.values(), b.values()):
    print(value1, value2)

>>>
1 red
2 blue
```

#### 기본 Dictionary

딕셔너리는 통계를 관리하고 추적하는 작업에 유용하다. 딕셔너리를 사용할 때 한 가지 문제는 어떤 키가 이미 존재한다고 가정할 수 없다는 점이다. 이 때문에 딕셔너리에 저장된 카운터를 증가시키는 것처럼 간단한 작업도 까다로워진다. 

```python
stats = {}
key = 'my_counter'
if key not in stats:
    stats[key] = 0
stats[key] += 1
```

collection 모듈의 defaultdict 클래스는 키가 존재하지 않으면 자동으로 기본값을 저장하도록 하여 이런 작업을 간소화한다. 할 일은 그저 키가 없을 때마다 기본값을 반환할 함수를 제공하는 것뿐이다. 다음 예제에서 내장 함수 int는 0을 반환한다(ref. 23. 인터페이스가 간단하면 클래스 대신 함수를 받자). 이제 카운터를 증가시키는 것은 간단한다.

```pyton
stats = defaultdict(int)
stats['my_counter'] += 1
```

#### 힙 큐

힙(heap)은 우선순위 큐(priority queue)를 유지하는 유용한 자료 구조다. heapq 모듈은 표준 list 타입으로 힙을 생성하는 heappush, heappop, nsmallest 같은 함수를 제공한다. 임의의 우선순위를 가지는 아이템을 어떤 순서로도 힙에 삽입할 수 있다.

```python
a = []
heappush(a, 5)
heappush(a, 3)
heappush(a, 7)
heappush(a, 4)
```

아이템은 가장 우선순위가 높은 것(가장 낮은 수)부터 제거된다.

```python
print(heappop(a), heappop(a), heappop(a), heappop(a))

>>>
3 4 5 7
```

결과로 만들어지는 list를 heapq 외부에서도 쉽게 사용할 수 있다. 힙의 0 인덱스에 접근하면 항상 가장 작은 아이템이 반환된다.

```python
a = []
heappush(a, 5)
heappush(a, 3)
heappush(a, 7)
heappush(a, 4)
assert a[0] == nsmallest(1, a)[0] == 3
```

list의 sort 메서드를 호출하더라도 힙의 불변성이 유지된다.

```python
print('Before:', a)
a.sort()
print('After:', a)

>>>
Before: [3, 4, 7, 5]
After: [3, 4, 5, 7]
```

이러한 각 heapq 연산에 걸리는 시간은 리스트의 길이에 비례하여 **로그** 형태로 증가한다. 표준 파이썬 리스트로 같은 동작을 수행하면 시간이 **선형**적으로 증가한다.

#### 바이섹션

list에서 아이템을 검색하는 작업은 index 메서드를 호출할 때 리스트의 길이에 비례한 선형적 시간이 걸린다.

```python
x = list(range(10**6))
i = x.index(991234)
```

bisect_left 같은 bisect 모듈의 함수는 정렬된 아이템 시퀀스를 대상으로한 효율적인 바이너리 검색을 제공한다. bisect_left가 반환한 인덱스는 시퀀스에 들어간 값의 삽입 지점이다.

```
i = bisect_left(x, 991234)
```

바이너리 검색의 복잡도는 **로그** 형태로 증가한다. 다시 말해 아이템 백만 개를 담은 리스트를 bisect로 검색할 때 걸리는 시간은 아이템 14개를 담은 리스트를 index로 순차 검색할 때 걸리는 시간과 거의 같다.

#### 이터레이터 도구

내장 모듈 itertools는 이터레이터를 구성하거나 이터레이터와 상호 작용하는 데 유용한 함수를 다수 포함한다(ref. 16, 17). 더 자세한 정보는 대화식 파이썬 세션에서 help(itertools)를 실행하면 볼 수 있다.

...

까다로운 이터레이션 코드를 다루는 상황이라면 itertools 문서를 살펴보고 쓸만한 기능이 있는지 찾아볼 가치가 있다.


* 알고리즘과 자료 구조를 표현하는 데는 파이썬의 내장 모듈을 사용하자.
* 이 기능들을 직접 재구현하지는 말자. 올바르게 만들기가 어렵기 때문이다.


## 47.

## 48. 

