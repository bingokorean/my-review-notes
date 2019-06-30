## 1. 파이썬다운 생각

'파이썬다운'이라는 형용사로 파이썬 스타일을 표현한다. 파이썬 스타일은 컴파일러가 정의하는 것이 아닌 파이썬 개발자들이 수년간 사용하면서 자연스럽게 생겨난 것이다. 복잡함보다는 단순함을, 가독성을 극대화하기 위해 명료한 것을 좋아한다. 

```
>>> import this
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

### 1. 파이썬 버전을 확인하자

```python
import sys
print(sys.version_info)
print(sys.version)
```

파이썬2에서 2to3, six와 같은 도구를 사용하면 파이썬3으로 쉽게 옮길 수 있다.

* 파이썬에는 CPython, Jython, IronPython, PyPy 같은 다양한 런타임 환경이 있다.
* 파이썬 커뮤니티에서 주로 다루는 버전은 파이썬3이므로 새 파이썬 프로젝트는 파이썬3가 좋다.

### 2. PEP 8 스타일 가이드를 따르자

파이썬 개선 제안서 [Python Enhancement Proposal #8 (PEP 8)](https://www.python.org/dev/peps/pep-0008/)를 참고하자. 일관성 있는 스타일로 유지보수가 더욱 쉬워지고 가독성도 높아지고 다양한 프로젝트에서 협업도 가능하다.

### 3. bytes, str, unicode 차이점을 알자

파이썬3에서는 bytes (raw 8비트; binary 데이터)와 str (unicode 문자) 두 가지 타입으로 문자 시퀀스를 나타낸다. bytes 인스턴스는 raw 8비트 값을 저장하고, str 인스턴스는 unicode 문자를 저장한다. unicode 문자를 binary 데이터로 '표현'하는 방법은 많다. (ex. UTF-8 인코딩). 허나, str 인스턴스는 연관된 binary 인코딩이 없다 (아예 변환하는 것). 따라서, unicode 문자를 binary 데이터로 '변환'하려면 encode 함수를, binary 데이터를 unicode 문자로 '변환'하려면 decode 함수를 사용해야 한다.

파이썬 프로그래밍할 때 외부에 제공할 인터페이스에서는 unicode를 encode하고 decode해야 한다 (즉, 바이너리 인코딩). 프로그램 핵심 부분에서는 unicode 문자 타입(ex. str)을 사용하고, 문자 인코딩에 대해서는 어떤 가정도 하지 말아야 한다. 그러기 위해서 다음 두 가지 헬퍼 함수가 필요하다.

```python
def to_str(bytes_or_str): # 파이썬 내부에서 사용될 때...
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value # str 인스턴스
    
def to bytes(bytes_or_str): # 외부에 보내질 때...
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
    return value # bytes 인스턴스
```

파이썬3에서 bytes와 str 인스턴스는 심지어 빈 문자열도 같지 않으므로 함수에 넘기는 문자열의 타입을 더 신중하게 처리해야 한다. 파이썬3에서 내장 함수 open이 반환하는 파일 핸들을 사용하는 연산은 기본적으로 UTF-8 인코딩을 사용한다. (파이썬2에서 파일 연산은 바이너리 인코딩이다) 따라서, 파이썬3에서 .bin 파일을 open할 때 'w'가 아닌 'wb'(바이너리-쓰기) 모드를, 'r'이 아닌 'rb'를 사용해야 한다.

* 파이썬3에서는 bytes는 8비트 값을 저장하고, str은 유니코드 문자를 저장한다. ('>'나 '+'와 같은 연산자에 bytes와 str 인스턴스를 함께 사용할 수 없다)
* 헬퍼 함수를 사용해서 처리할 입력값이 원하는 문자 시퀀스 타입(8비트 값, UTF-8 인코딩 문자, 유니코드 문자 등)으로 되어 있게 한다.
* 바이너리 데이터를 파일에서 읽거나 쓸 때는 파일을 바이너리 모드('rb' 혹은 'wb')로 오픈한다.

### 4. 복합한 표현식 대신 헬퍼 함수를 사용하자

파이썬은 간결한 문법을 이용하면 많은 로직을 표현식 한 줄로 쉽게 작성할 수 있다. 예를 들어 URL에서 쿼리 문자열을 디코드해야 한다고 하자. 다음 예에서 각 쿼리 문자열 파라미터는 정수 값을 표현한다.

```python
from urllib.parse import parse_qs
my_values = parse_qs('red=5&blue=0&green=', keep_blank_values=True)
```

쿼리 문자열 파라미터에 따라 값이 여러 개 있을 수도 있고 값이 한 개만 있을 수도 있으며, 파라미터는 존재하지만 값이 비어 있을 수 있고, 파라미터가 아예 빠진 경우도 있다. 파라미터가 없거나 비어 있으면 기본값으로 0을 할당하면 좋다. 다음 처리 방식을 보자.

```python
red = my_values.get('red', [''])[0] or 0  # '5'
green = my_values.get('green', [''])[0] or 0 # 0
opacity = my_values.get('opacity', [''])[0] or 0 # 0
```

숫자 변환을 위해 `red = int(my_values.get('red', [''])[0] or 0)`으로 할 수도 있다. 이들의 코드를 읽기는 쉽지 않다. if/else 문이 훨씬 더 직감적일 것이다.

```python
green = my_values.get('green', [''])
if green[0]:
    green = int(green[0])
else:
    green = 0
```

하지만 이를 반복적으로 사용하기 보다는 다음과 같은 헬퍼 함수로 처리하면 어떨까?

```python
def get_first_int(values, key, default=0):
    found = values.get(key, [''])
    if found[0]:
        found = int(found[0])
    else:
        found = default
    return found
```

`green = get_first_int(my_values, 'green')` 이렇게 헬퍼 함수를 이용하면, or를 사용한 복잡한 표현식이나 if/else 조건식 버전보다 호출 코드가 훨씬 더 간결하고 명확해진다. 표현식이 복잡해지기 시작하면 최대한 빨리 해당 표현식을 작은 조각으로 분할하고 로직을 헬퍼 함수로 옮기는 방안을 고려해야 한다. 무조건 짧은 코드를 만들기보다는 가독성을 선택하는 편이 낫다. 이해하기 어려운 복잡한 표현식에는 파이썬의 함축적인 문법을 사용하면 안 된다.

* 파이썬의 문법은 한 줄짜리 표현식을 쉽게 작성할 수 있지만 코드가 복잡해지고 읽기 어려워진다.
* 복잡한 표현식은 헬퍼 함수로 옮기는 게 좋다. 특히, 같은 로직을 반복해서 사용한다면 헬퍼 함수를 사용하자.
* if/else 표현식을 이용하면 or나 and 같은 불 연산자를 사용할 때보다 읽기 수월한 코드를 작성할 수 있다.

### 5. 시퀀스를 슬라이스하는 방법을 알자

파이썬은 시퀀스를 slice해서 부분집합에 접근할 수 있도록 해준다. 가장 간단한 슬라이싱 대상은 내장 타입인 list, str, btyes이다. `__getitem__`과 `__setitem__`이라는 특별한 메서드를 구현하는 클래스에도 slicing을 적용할 수 있다. slicing 기본 문법 형태는 `somelist[start:end]`이며, 여기서 start 인덱스는 포함되고 end 인덱스는 제외된다.

list의 처음부터 slice할 때는 보기 편하게 인덱트 0을 생략하고, list의 끝까지 slice할 때도 마지막 인덱스는 넣지 않아도 되므로 생략한다.

```python
assert a[:5] == a[0:5]
assert a[5:] == a[5:len(a)]
```

리스트의 끝을 기준으로 오프셋을 계산할 때는 음수로 slice하는 게 편하다. 

```python
a = [1,2,3,4,5,6,7,8,9]
a[-4:] # 뒤에서 4번째까지 추출해라.
>>> [6,7,8,9] 
a[-4:-1] # 이런 형태는 조심
>>> [6,7,8]
```

slicing은 start와 end 인덱스가 리스트의 경계를 벗어나도 적절하게 처리한다. 덕분에 입력 시퀀스에 대응해 처리할 최대 길이를 코드로 쉽게 설정할 수 있다. 이와 대조적으로 같은 인덱스를 직접 접근하면 예외가 발생한다.

```python
first_twenty_items = a[:20]
>>> [1,2,3,4,5,6,7,8,9]
a[20]
>>> IndexError: list index out of range
```

[NOTE] 리스트의 인덱스를 음수로 지정하면 slicing이 뜻밖의 결과를 얻기도 한다. 예를 들어, `somelist[-n:]`이라는 구문은 `somelist[-3:]`처럼 n이 1보다 클 때는 제대로 동작하지만, n이 0이어서 `somelist[-0:]`이 되면 원본 리스트의 복사본을 만든다. slicing 결과는 완전히 새로운 리스트이지만, 원본 리스트에 들어 있는 객체에 대한 참조는 유지된다. 하지만, slice한 결과를 수정해도 원본 리스트에 아무런 영향을 미치지 않는다. 

할당에 사용하면 slice는 원본 리스트에서 지정한 범위를 대체한다. `a, b = c[:2]` 같은 튜플 할당과 달리 slice 할당은 길이가 달라도 된다. 할당받은 slice 앞뒤 값은 유지된다. 리스트는 새로 들어온 값에 맞춰 늘어나거나 줄어든다.

```python
a[2:7] = [99,22,14]
print(a)
>>> [1,2,99,22,14,8,9]
```

시작과 끝 인덱스를 모두 생략하고 slice하면 원본 리스트의 복사본을 얻는다
```python
b = a[:]
assert b == a and b is not a
```

slice에 시작과 끝 인덱스를 지정하지 않고 할당하면 (새 리스트를 할당하지 않고) slice의 전체 내용을 참조 대상의 복사본으로 대체한다. (즉, 주소 복사 실시)

```python
b = a
a[:] = [101,102,103]
print(a)
>>> [101,102,103]
print(b)
>>> [101,102,103]
```

* start 인덱스에 0을 설정하거나 end 인덱스에 시퀀스의 길이를 설정하지 말자. 
* slicing은 범위를 벗어난 start나 end 인덱스를 허용하므로 a[:20]이나 a[-20]처럼 시퀀스의 앞쪽이나 뒤쪽 경계에 놓인 slice를 표현하기 쉽다.
* list slice에 할당하면 원본 시퀀스에 지정한 범위를 참조 대상의 내용으로 대체한다 (길이가 달라도 동작; 즉, 주소참조를 주의하자)


### 6. 한 슬라이스에 start, end, stride를 함께 쓰지 말자

파이썬에는 기본 slicing과 `somelist[start:end:stride]` 처럼 slice에 stride를 설정하는 특별한 문법도 있다. 이 문법을 이용하면 시퀀스를 slice할 때 매 n번째 아이템을 가져올 수 있다. 

```python
a = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
odds = a[::2]
evens = a[1::2]
print(oods)
print(evens)
>>> ['red', 'yellow', 'blue']
>>> ['orange', 'green', 'purple']
```

문제는 stride 문법이 종종 예상치 못한 동작을 해서 버그를 만들어내기도 한다. 예를 들어, 파이썬에서 바이트 문자열을 역순으로 만든느 일반적인 방법은 stride -1로 문자열을 slice하는 것이다. 문제는 바이트 문자열이나 아스키 문자에는 잘 동작하지만, UTF-8 바이트 문자열로 인코드된 유니코드 문자에는 원하는 대로 동작하지 않는다. 

```python
w = '漢字'
x = w.encode('utf-8')
y = x[::-1]
z = y.decode('utf-8')
>>> 
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9d in 
position 0: invalid start byte
```

-1을 제외한 음수 값으로 stride를 지정하면 어떨까? 2::2는 무슨 뜻일까? 요점은 slicing 문법의 stride 부분이 매우 혼란스러울 수 있다는 점이다. 대괄호 안에 숫자가 세 개나 있으면 빽빽해서 읽기 어렵고 start와 end 인덱스가 stride와 연계되어 어떤  작용을 하는지 분명하지 않다. 특히 stride가 음수인 경우는 더욱 그러하다.

이러한 문제를 방지하기 위해 stride를 start, end 인덱스와 함께 사용하지 말아야 한다. stride를 사용해야 한다면 양수 값을 사용하고 start와 end 인덱스는 생략하는 게 좋다. stride를 꼭 start와 end 인덱스와 함께 사용해야 한다면 stride를 적용한 결과를 변수에 할당하고, 이 변수를 slice한 결과를 다른 변수에 할당해서 사용하자.

```python
a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
b = a[::2] # ['a', 'c', 'e', 'g']
c = b[1:-1]
```

slicing부터 하고 striding을 하면 데이터의 shallow copy가 추가로 생긴다. 첫 번째 연산은 결과로 나오는 slice 크기를 최대한 줄여야 한다. 프로그램에서 두 과정이 필요한 시간과 메모리가 충분하지 않다면 내장 모듈 itertools의 islice 메서드를 사용하자. islice 메서드는 start, endm, stride에 음수 값을 허용하지 않는다.

* 한 slice에 start, end, stride를 지정하면 혼란스러울 수 있다.
* slice에 start와 end 인덱스 없이 양수 stride 값을 사용하자. 음수 stride 값은 가능하면 피하는 게 좋다.
* 한 slice에 start, end, stride를 함께 사용하는 상황은 피하자. 파라미터 세 개를 사용해야 한다면 두 개(slice, 다른 하나는 stride)를 사용하거나 내장 모듈 itertools의 islice를 사용하자.

### 7. map과 filter 대신 list comprehension을 사용하자

파이썬은 list comprehension(리스트 함축 표현식)을 통해 한 리스트에서 다른 리스트로 간결하게 만들 수 있다. 예를 들어 리스트에 있는 각 숫자의 제곱 계산은 다음과 같다.

```python
a = [1,2,3,4,5,6,7,8,9,10]
squares = [x**2 for x in a]
print(squares)
>>>
[1,4,9,16,25,36,49,64,81,100]
```

인수가 하나뿐인 함수를 적용하는 상황이 아니면, 간단한 연산에는 list comprehension이 내장 함수 map보다 명확하다. map은 계산에 필요한 lambda 함수를 생성해야 해서 깔끔하지 않다.

```python
squares = map(lambda x: x ** 2, a)
```

list comprehension은 map과 달리 입력 리스트에 있는 아이템을 간편하게 걸러내서 그에 대응하는 출력을 결과에서 삭제할 수 있다. 예를 들어 2로 나누어 떨어지는 숫자의 제곱만 계산한다고 하자.

```python
even_squares = [x**2 for x in a if x % 2 == 0]
print(even_squares)
>>>
[4,16,36,64,100]
```

내장 함수 filter를 map과 연계해서 사용해도 같은 결과를 얻지만 훨씬 읽기 어렵다.

```python
alt = map(lambda x: x**2, filter(lambda x: x % 2 ==0, a))
assert even_squares == list(alt)
```
dictionary와 set에도 list comprehension에 해당하는 문법이 있다. comprehension 문법을 쓰면 알고리즘을 작성할 때 파생되는 자료 구조를 쉽게 생성할 수 있다. 

```python
chile_ranks = {'ghost':1, 'habanero':2, 'cayenne':3}
rank_dict = {rank: name for name, rank in chile_ranks.items()}
chile_len_set = {len(name) for name in rank_dict.values()}
print(rank_dict)
print(chile_len_set)
>>>
{1: 'ghost', 2: 'habanero', 3: 'cayenne'}
{8, 5, 7}
```

* list comprehension은 추가적인 lambda 표현식이 필요 없어서 내장 함수인 map이나 filter를 사용하는 것보다 명확하다
* list comprehension을 사용하면 입력 리스트에서 아이템을 간단히 건너뛸 수 있다. map으로는 filter를 사용하지 않고서는 이런 작업을 못한다
* dictionary와 set도 comprehension 표현식을 지원한다

### 8. List comprehension에서 표현식을 두 개 넘게 쓰지 말자

List comprehension은 기본 사용법(ref.7)뿐만 아니라 다중 루프도 지원한다. 예를 들어, 행렬을 평평한 리스트 하나로 간략화해보자. 

```python
matrix = [[1,2,3], [4,5,6], [7,8,9]]
flat = [x for row in matrix for x in row]
print(flat)
>>>
[1,2,3,4,5,6,7,8,9]
```

다중 루프의 또 다른 사용법은 입력 리스트의 레이아웃을 두 레벨로 중복해서 구성하는 것이다. 예를 들어 2차원 행렬의 각 셀에 있는 값의 제곱을 구한다고 하자. 이 표현식은 추가로 [] 문자를 사용하므로 그리 좋아 보이진 않지만 이해하기는 쉽다.

```python
squared = [[x**2 for x in row] for row in matrix]
print(squared)
>>>
[[1,4,9], [16,25,36], [49,64,81]]
```

이 표현식을 다른 루프에 넣는다면 list comprehension이 여러 줄로 구분해야 할 정도로 길어진다.

```python
my_lists = [
    [[1,2,3], [4,5,6]],
    #...
]
flat = [x for sublist1 in my_lists
        for sublist2 in sublist1
        for x in sublist2]
```

이 경우는 일반 루프문으로 들여쓰기를 사용하면 list comprehension보다 이해하기 더 쉽다.

```python
flat = []
for sublist1 in my_lists:
    for sublist2 in sublist1:
        flat.extend(sublist2)
```

List comprehension도 다중 if 조건을 지원한다. 같은 루프 레벨에서 여러 조건이 있으면 암시적인 and 표현식이 된다. 예를 들어 숫자로 구성된 리스트에서 4보다 큰 짝수 값만 가지고 온다면 다음 두 list comprehension은 동일하다. 조건은 루프의 각 레벨에서 for 표현식 뒤에 설정할 수 있다.

```python
a = [1,2,3,4,5,6,7,8,9,10]
b = [x for x in a if x > 4 if x % 2 ==0 ]
c = [x for x in a if x > 4 and x % 2 == 0]
```

문제는 행렬에서 if 조건이 들어갈 경우 list comprehension으로 간략히 표현할 수 있지만 이해하기 매우 어렵다.

```python
matrix = [[1,2,3], [4,5,6], [7,8,9]]
filtered = [[x for x in row if x % 3 == 0]
            for row in matrix if sum(row) >= 10]
print(filtered)
>>>
[[6], [9]]
```

경험에 비추어 볼 때 list comprehension을 사용할 때는 표현식이 두 개를 넘어가면 피하는 게 좋다. 조건 두 개, 루프 두 개, 혹은 조건 한 개와 루프 한 개 정도면 된다. 이거솝다 복잡해지면 일반적인 if문과 for문을 사용하고 헬퍼 함수(16.참조)를 작성하자.

* List comprehension은 다중 루프와 루프 레벨별 다중 조건을 지원한다.
* 표현식이 두 개가 넘게 들어 있는 list comprehension은 이해하기 매우 어려우므로 피해야 한다.


### 9. Comprehension이 클 때는 generator 표현식을 고려하자

List comprehension의 문제점(7.참고)은 입력 시퀀스에 있는 각 값별로 아이템을 하나씩 담은 새 리스트를 통째로 생성한다는 점이다. 입력이 적을 때는 괜찮지만 클 때는 메모리를 많이 소모해 프로그램을 망가뜨리는 원인이 될 수 있다. 예를 들어, 파일을 읽고 각 줄에 있는 문자의 개수를 반환한다고 하자. List comprehension으로 하면 파일에 있는 각 줄의 길이만큼 메모리가 필요하다. 특히, 파일에 오류가 있거나 끊김이 없는 네크워트 소켓일 경우 list comprehension을 사용하면 문제가 발생한다. 

```python
value = [len(x) for x in open('/tmp/my_file.txt')]
print(value)
>>>
[100, 57, 15, 1, 12, 75, 5, 86, 89, 11]
```

파이썬은 이 문제를 해결하기 위해서 list comprehension과 generator를 일반화한 generator extpression을 제공한다. Generator expression은 실행될 때 출력 시퀀스를 모두 메모리에 로딩하지 않는다. 대신에 expression에서 한 번에 한 아이템을 내주는 iterator로 평가되고, () 문자 사이의 문법으로 표현된다.

```python
it = (len(x) for x in open('/tmp/my_file.txt'))
print(it)
>>>
<generator object <genexpr> at 0x101b81480>     # 값이 아니라 주소를 바라보고 있으므로, 주소를 출력하는 것 같다.
```

출력을 생성하기 위해서는 내장 함수 next로 반환받은 iterator를 한 번에 전진시키면 된다. 이로써 코드에서 메모리 사용량을 걱정하지 않고 geneartor expression을 사용하면 된다.

```python
print(next(it))
print(next(it))
>>>
100
75
```

Generator expression의 또 다른 강력한 기능은 다른 geneartor expression과 함께 사용할 수 있다는 점이다. 

```python
root = ((x, x**0.5) for x in it)
print(next(roots))
>>>
(15, 3.872983346207417)
```

이 iterator를 전진시킬 때마다 루프의 도미노 효과로 내부 iterator도 전진시키고 조건 표현식을 계산해서 입력과 출력을 처리한다.

이처럼  generator를 연결하면 파이썬에서 매우 빠르게 실행할 수 있다. 큰 입력 스트림에 동작하는 기능을 결합하는 방법을 찾을 때는 generator expression이 최선의 도구다. 단, generator expression이 반환한 iterator에는 상태가 있으므로 iterator를 한 번 넘게 사용하지 않도록 주의해야 한다(ref.17)

* List comprehension은 큰 입력을 처리할 때 너무 많은 메모리를 소모해서 문제를 일으킬 수 있다. Generator expression은 iterator로 한 번에 한 출력만 만드므로 메모리 문제를 피할 수 있다.
* 한 generator expression에서 나온 iterator를 또 다른 generator expression의 for 서브 expression으로 넘기는 방식으로 geneator expression을 조합할 수 있다.
* Generator expression은 서로 연결되어 있을 때 매우 빠르게 실행된다.


### 10. range보다는 enumerate를 사용하자

내장 함수 range는 정수 집합을 순회(iterate)하는 루프를 실행할 때 유용하다.

```python
random_bits = 0
for i in range(64):
    if randint(0, 1):
        random_bits |= 1 << i
```

문자열의 리스트 같이 순회할 자료 구조가 있을 때는 직접 루프를 실행할 수 있다. 리스트를 순회하거나 리스트의 현재 아이템의 인덱스를 알고 싶은 경우가 있을 떄 range를 사용하면 된다.

```python
flavor_list = ['vanilla', 'chocolate', 'pecan', 'strawberry']
for flavor in flavor_list:
    print('%s is delicious' % flavor)
	
for i in range(len(flavor_list)):
    flavor = flavor_list[i]
    print('%d %d' % (i +1, flavor))
```

위의 코드는 약간 세련되지 못하다. 리스트의 길이를 알아내야 하고, 배열을 인덱스로 접근해야 하며 읽기 불편하다. 파이썬은 이런 상황을 처리하기 위해 내장 함수 enumerate를 제공한다 enumerate는 lazy generator로 iterator를 감싼다. 

```
for i, flavor in enumerate(flavor_list):
    print('%d: %s' % (i+1, flavor))
```

enumerate를 세기 시작할 숫자를 지정하면 코드를 더 짧게 만들 수 있다.

```
for i, flavor in enumerate(flavor_list, 1):
    print('%d: %s' % (i+1, flavor))
```

* enumerate는 iterator를 순회하면서 iterator에서 각 아이템의 인덱스를 얻어오는 간결한 문법을 제공한다
* range로 루프를 실행하고 시퀀스에 인덱스로 접근하기보다는 enumerate를 사용하는 게 좋다
* enumerate에 두 번째 파라미터를 사용하면 세기 시작할 숫자를 지정할 수 있다 (기본값은 0)


### 11. Iterator를 병렬로 처리하려면 zip을 사용하자

파이썬에서 리스트 객체를 많이 사용한다. List comprehension을 사용하면 source list에 표현식을 적용하여 derived list를 쉽게 얻는다 (ref.7). 보통 derived list와 source list는 연관되어 있다. 두 리스트를 병렬로 순회하기 명료한 방법은 무엇일까? 내장 함수 zip을 사용하는 것이다.

파이썬3에서 zip은 지연 generator로 두 개 이상의 iterator를 감싼다. zip generator는 각 iterator로부터 다음 값을 담은 튜플을 얻어온다. zip generator를 사용한 코드는 다중 리스트에서 인덱스로 접근하는 코드보다 훨씬 명료하다.

```python
name = ['Cecilia', 'Lise', 'Marie']
letters = [len(n) for n in names]
for name, count in zip(names, letters):
    if count > max_letters:
        longest_name = name
        max_letters = count
```

내장 함수 zip의 문제점은 입력 iterator들의 길이가 다르면 zip이 이상하게 동작한다는 점이다. 즉, 최소 길이까지만 동작한다. zip으로 실행할 리스트의 길이가 같다고 확신할 수 없으면 내장 모듈 itertools의 zip_longest를 사용하는 방안을 고려하자.

* 내장 함수 zip은 여러 iterator를 병렬로 순회할 때 사용하고, 튜플을 생성하는 지연 iterator이다.
* 길이가 다른 iterator를 사용하면 zip은 그 결과를 최소길이를 기준으로 잘라낸다.
* 내장 모듈 itertools의 zip_longest 함수를 쓰면 여러 iterator 길이에 상관없이 병렬로 순회할 수 있다 (46.참고)


### 12. for와 while 루프 뒤에는 else 블록을 쓰지 말자

파이썬의 루프에는 다른 프로그래밍 언어에는 없는 추가적인 기능이 있는데, 루프에서 반복되는 내부 블록 바로 다음에 else 블록을 둘 수 있는 기능이다. 

```python
for i in range(3):
   print('Loop %d' % i)
   #if i == 1:
   #   break
else:
   print('Else block')
>>>
Loop 0
Loop 1 
Loop 2
Else block!
```

놀랍게도 else 블록은 루프가 종료되면 실행된다. 또는, 루프에서 break문을 사용하면 else 블록으로 건너뛴다. 다소 놀랄 만한 점은 빈 시퀀스를 처리하는 루프문에서도 else 블록이 즉시 실행된다. 그리고 while False와 같이 처음부터 거짓인 경우에도 실행된다. 

이렇게 동작하는 이유는 루프 다음에 오는 else 블록을 루프로 뭔가를 검색할 때 유용하기 때문이다. 예를 들어, 서로소(coprime; 공약수가 1밖에 없는 둘 이상의 수)를 판별한다고 하자. 

```python
a = 4
b = 9
for i in range(2, min(a,b) + 1):
    print('Testing', i)
    if a % i == 0 and b % i ==0:
        print('Not coprime')
        break
else:
    print('Coprime')
>>>
Testing 2
Testing 3
Testing 4
Coprime
```

실제로 이런 방식으로 코드를 작성하면 안 된다. 대신에 이런 계산을 하는 헬퍼 함수를 작성하는 게 좋다. 이런 헬퍼 함수는 두 가지 일반적인 스타일로 작성할 수 있다. 이 방법으로 낯선 코드를 접하는 개발자들이 코드를 훨씬 쉽게 이해할 수 있다. 

첫 번째는 찾으려는 조건을 찾았을 때 바로 반환하는 것이다. 루프가 실패로 끝나면 기본 결과(True)를 반환한다.

```python
def coprime(a, b):
    for i in range(2, min(a,b)+1):
        if a % i == 0 and b % i == 0:
            return False
    return True
```

두 번째는 루프에서 찾으려는 대상을 찾았는지 알려주는 결과 변수를 사용하는 것이다. 뭔가를 찾으면 즉시 break로 루프를 중단한다.

```python
def coprime2(a, b):
    is_coprime = True
    for i in range(2, min(a, b) + 1):
        if a % i == 0 and b % i == 0:
            is_coprime = False
            break
    return is_coprime
```

루프처럼 간단한 구조는 파이썬에서 따로 설명할 필요가 없어야 한다. 그러므로 루프 다음에 오는 else 블록은 절대로 사용하지 말아야 한다.
* 파이썬에는 for와 while 루프의 내부 블록 바로 뒤에 else 블록을 사용할 수 있게 하는 특별한 문법이 있다.
* 루프 본문이 break문을 만나지 않은 경우에만 루프 다음에 오는 else블록이 실행된다.
* 루프 뒤에 else 블록을 사용하면 직관적이지 않고 혼동하기 쉬우니 사용하지 말아야 한다.


### 13. try/except/else/finally에서 각 블록의 장점을 이용하자

파이썬의 예외 처리 과정은 try, except, else, finally 블록으로 각 시점을 처리한다. 각 블록은 복합문에서 독자적인 목적이 있으며, 이 블록들을 다양하게 조합하면 유용하다 (또 다른 예로 51 참고).

[finally 블록] 예외를 전달하고 싶지만, 예외가 발생해도 정리 코드를 실행하고 싶을 때 try/finally를 사용하면 된다. try/finally의 일반적인 사용 예 중 하나는 파일 핸들러를 제대로 종료하는 작업이다 (또 다른 접근법으로 43 참고). 파일을 열 때 일어나는 예외는 finally 블록에서 처리하지 않아야 하므로 try 블록 앞에서 open을 호출해야 한다.

```python
handle = open('/tmp/random_data.txt')   # IOError가 일어날 수 있음
try:
    data = handle.read()    # UnicodeDecodeError가 일어날 수 있음
finally:
    handle.close()    # try: 이후에 항상 실행됨
```

[else 블록] 코드에서 어떤 예외를 처리하고 어떤 예외를 전달할지를 명확하게 하려면 try/except/else를 사용해야 한다. try 블록이 예외를 일으키지 않으면 else 블록이 실행된다. else 블록을 사용하면 try 블록의 코드를 최소로 줄이고 가독성을 높일 수 있다.

```python
def load_json_key(data, key):
    try:
        result_dict = json.loads(data)    # ValueError가 일어날 수 있음
    except ValueError as e:
        raise KeyError from e
    else:
        return result_dict[key]    # KeyError가 일어날 수 있음
```

else 절은 try/except 다음에 나오는 처리를 시각적으로 except 블록과 구분해준다. 그래서 예외 전달 행위를 명확하게 한다.

[모두 함께 사용] 복합문 하나로 모든 것을 처리하고 싶다면 try/except/else/finally를 사용하면 된다.

```python
UNDEFINED = object()

def divide_json(path):
    handle = open(path, 'r+')    # IOError가 일어날 수 있음
    try:
        data = handle.read()    # UnicodeDecodeError가 일어날 수 있음
        op = json.loads(data)    # ValueError가 일어날 수 있음
        value = (
            op['numerator'] \
            op['denominator'])    # ZeroDivisionError가 일어날 수 있음
    except ZeroDivisionError as e:
        return UNDEFINED
    else:
        op['result'] = value
        result = json.dumps(op)
        handle.seek(0)
        handle.write(result)    # IOError가 일어날 수 있음
        return value
    finally:
        handle.close()    # 항상 실행함
```

* try/finally 복합문을 이용하면 try블록에서 예외 발생 여부와 상관없이 정리 코드를 실행할 수 있다.
* else 블록은 try 블록에 있는 코드의 양을 최소로 줄이는 데 도움을 주며 try/except 블록과 성공한 경우에 실행할 코드를 시각적으로 구분해준다.
* else 블록은 try 블록의 코드가 성공적으로 실행된 후 finally 블록에서 공통 정리 코드를 싱행하기 전에 추가 작업을 하는 데 사용할 수 있다.

