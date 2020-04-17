# Effective PYTHON - 파이썬 코딩의 기술

2016.03.31, 브렛 슬라킨 지음, 김형철 옮김

### Contents

<div id='contents'/>

1. [파이썬다운 생각](#1.)
2. [함수](#2.)
3. [클래스와 상속](#3.)
4. 메타클래스와 속성
5. [병행성과 병렬성](#5.)
6. [내장 모듈](#6.)
7. [협력](#7.)
8. [제품화](#8.)

<br>

<div id='1.'/>

## 1장. 파이썬다운 생각

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
red = my_values.get('red', [''])[0] or 0            # '5'
green = my_values.get('green', [''])[0] or 0        # 0
opacity = my_values.get('opacity', [''])[0] or 0    # 0
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
```
```
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
```
```
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
```
```
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
```
```
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
```
```
>>>
[1,2,3,4,5,6,7,8,9]
```

다중 루프의 또 다른 사용법은 입력 리스트의 레이아웃을 두 레벨로 중복해서 구성하는 것이다. 예를 들어 2차원 행렬의 각 셀에 있는 값의 제곱을 구한다고 하자. 이 표현식은 추가로 [] 문자를 사용하므로 그리 좋아 보이진 않지만 이해하기는 쉽다.

```python
squared = [[x**2 for x in row] for row in matrix]
print(squared)
```
```
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
```
```
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
```
```
>>>
[100, 57, 15, 1, 12, 75, 5, 86, 89, 11]
```

파이썬은 이 문제를 해결하기 위해서 list comprehension과 generator를 일반화한 generator extpression을 제공한다. Generator expression은 실행될 때 출력 시퀀스를 모두 메모리에 로딩하지 않는다. 대신에 expression에서 한 번에 한 아이템을 내주는 iterator로 평가되고, () 문자 사이의 문법으로 표현된다.

```python
it = (len(x) for x in open('/tmp/my_file.txt'))
print(it)
```
```
>>>
<generator object <genexpr> at 0x101b81480>     # 값이 아니라 주소를 바라보고 있으므로, 주소를 출력하는 것 같다.
```

출력을 생성하기 위해서는 내장 함수 next로 반환받은 iterator를 한 번에 전진시키면 된다. 이로써 코드에서 메모리 사용량을 걱정하지 않고 geneartor expression을 사용하면 된다.

```python
print(next(it))
print(next(it))
```
```
>>>
100
75
```

Generator expression의 또 다른 강력한 기능은 다른 geneartor expression과 함께 사용할 수 있다는 점이다. 

```python
root = ((x, x**0.5) for x in it)
print(next(roots))
```
```
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
```
```
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
```
```
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


<br>

<div id='2.'/>

## 2장. 함수

함수는 큰 프로그램을 작고 단순한 조각으로 나눌 수 있게 해준다. 함수를 사용하면 가독성이 높아지고 코드가 더 이해하기 쉬워진다. 또한 재사용이나 리팩토링도 가능해진다. 파이썬에서 제공하는 함수들에는 부가 기능이 있는데 이는 함수의 목적을 더 분명하게 하고, 불필요한 요소를 제거하고 호출자의 의도를 명료하게 보여주며, 찾기 어려운 미묘한 버그를 상당수 줄여줄 수 있다.

### 14. None을 반환하기보다는 예외를 일으키자

파이썬 프로그래머들은 유틸리티 함수를 작성할 때 반환 값 None에 특별한 의미를 부여하는 경향이 있다. 예를 들어, 어떤 숫자를 다른 숫자로 나누는 헬퍼 함수의 경우, 0으로 나누는 경우에는 결과가 정의되어 있지 않으므로 None을 반환하는 게 자연스럽다.

```python
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

result = divide(x, y)
if result is None:
    print('Invalid inputs')
    
x, y = 0, 5
result = divide(x, y)
if not result:
    print('Invalid inputs')
```

문제는 분자가 0이 되는 경우 반환 값도 0이 된다는 점인데, if문과 같은 조건에서 결과를 평가할 때 None 대신에 False에 해당하는 값을 검사할 수 있다 (참고 4). 이러한 오류를 줄이는 좋은 방법은 절대로 None을 반환하지 않고, 호출하는 쪽에 예외를 일으켜서 그 예외를 처리하게 하는 것이다.

```python
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        raise ValueError('Invalid inputs') from e
      
x, y = 5, 2
try:
    result = divide(x, y)
except ValueError:
    print('Invalid inputs')
else:
    print('Result is %.1f' % result)
```
```
>>> 
Result is 2.5
```

호출하는 쪽에서는 잘못된 입력에 대한 예외를 처리해야 한다 (49 참고). 호출하는 쪽에서 더는 함수의 반환 값을 조건식으로 검사할 필요가 없다. 함수가 예외를 일으키지 않았다면 반환 값은 문제가 없다. 예외를 처리하는  코드도 깔끔해진다.

* 특별한 의미를 나타내려고 None을 반환하는 함수가 오류를 일으키기 쉬운 이유는 None이나 다른 값(예를 들면 0이나 빈 문자열)이 조건식에서 False로 평가되기 때문이다.
* 특별한 상황을 알릴 때 None을 반환하는 대신에 예외를 일으키자. 문서화가 되어 있다면 호출하는 코드에서 예외를 적절하게 처리할 것이라고 기대할 수 있다.

### 15. Closure가 변수 scope와 상호 작용하는 방법을 알자

숫자 리스트를 정렬할 때 특정 그룹의 숫자들이 먼저 오도록 우선순위를 매기려고 한다. 이런 패턴은 사용자 인터페이스를 표현하거나 다른 것보다 중요한 메시지나 예외 이벤트를 먼저 보여줘야 할 때 유용하다. 이렇게 만드는 일반적인 방법은 리스트의 sort 메서드에 헬퍼 함수를 key 인수로 넘기는 것이다. 헬퍼의 반환 값은 리스트에 있는 각 아이템을 정렬하는 값으로 사용된다. 헬퍼는 주어진 아이템이 중요한 그룹에 있는지 확인하고 그에 따라 정렬 키를 다르게 할 수 있다.

```python
def sort_priority(values, group):
    def helper(x):
        if x in group:
            return (0, x)
        return (1, x)
    values.sort(key=helper)

numbers = [8,3,1,2,5,4,7,6]
group = {2,3,5,7}
sort_priority(numbers, group)
print(numbers)
```
```
>>>
[2,3,5,7,1,4,6,8]
```

위와 같이 동작한 이유는 다음과 같다.
* 파이썬은 자신이 정의된 scope에 있는 변수를 참조하는 함수인 closure를 지원한다. 이 때문에 helper 함수가 sort_priority의 group 함수에 접근할 수 있다.
* 함수는 파이썬에서 first-class object이다. 이 말은 함수를 직접 참조하고, 변수에 할당하고, 다른 함수의 인수로 전달하고, 표현식과 if문 등에서 비교할 수 있다는 의미다. 따라서, sort 메서드에서 closure 함수를 key 인수로 받을 수 있다. 
* 파이썬에는 tuple을 비교하는 특정한 규칙이 있다. 먼저 index 0으로 아이템을 비교하고 그 다음으로 index 1, 다음은 index 2와 같이 진행한다. helper closure의 반환 값이 정렬 순서를 분리된 두 그룹으로 나뉘게 한 건 이 규칙때문이다.

추가적으로, 위 함수에서 우선순위가 높은 아이템을 발견했는지 여부를 반환해서 사용자 인터페이스 코드가 그에 따라 동작하게 하면 좋을 것이다.

```python
def sort_priority2(numbers, group):
   found = False                # scope: 'sort_priority2'
   def helper(x):
       if x in group:
           found = True         # scope: 'helper' -- 안 좋음!!
           return (0, x)
       return (1, x)
   numbers.sort(key=helper)
   return found

found  = sort_priority2(numbers, group)
print('Found:', found)
print(numbers)
```
```
>>>
Found: false
[2,3,5,7,1,4,6,8]
```

정렬된 결과는 올바르지만 found 결과는 틀렸다. group에 속한 아이템을 numbers에서 찾을 수 있었지만 함수는 False를 반환했다. 어째서 이런 일이 일어났을까?

표현식에서 변수를 참조할 때 파이썬 인터프리터는 참조를 해결하려고 다음과 같은 순서로 scope(유효 범위)를 탐색한다. 다음 중 어느 scope에도 참조한 이름으로 된 변수가 정의되어 있지 않으면 NameError 예외가 일어난다.
1. 현재 함수의 scope
2. (현재 scope를 담고 있는 다른 함수 같은) 감싸고 있는 scope
3. 코드를 포함하고 있는 모듈의 scope (전역 scope라고도 함)
4. (len이나 str같은 함수를 담고 있는) 내장 scope

변수에 값을 할당할 때는 다른 방식으로 동작한다. 변수가 이미 현재 scope에 정의되어 있다면 새로운 값을 얻는다. 파이썬은 변수가 현재 scope에 존재하지 않으면 변수 정의로 취급한다. 새로 정의되는 변수의 scope는 그 할당을 포함하고 있는 함수가 된다. 이 할당 동작은 sort_priority2 함수의 반환 값이 잘못된 이유를 설명한다. found 변수는 helper closure에서 True로 할당된다. Closure 할당은 sort_priority2에서 일어나는 할당이 아닌 helper 안에서 일어나는 새 변수 정의로 처리된다. 

이 문제는 scoping bug라고도 불리나 언어 설계자가 의도한 결과이다. 이 동작은 함수의 지역 변수가 자신을 포함하는 모듈을 오염시키는 문제를 막아준다. 그렇지 않았다면 함수 안에서 일어나는 모든 할당이 전역 모듈 scope에 쓰레기를 넣는 결과로 이어졌을 것이다. 그렇게 되면 불필요한 할당에 그치지 않고 결과로 만들어지는 전역 변수들의 상호 작용으로 알기 힘든 버그가 생긴다.

파이썬3에는 closure에서 데이터를 얻어오는 특별한 문법이 있다. nonlocal문은 특정 변수 이름에 할당할 때 scope 탐색이 일어나야 함을 나타낸다. 유일한 제약은 nonlocal (전역 변수의 오염을 피하려고) 모듈 수준 scope까지는 탐색할 수 없다는 점이다. 

```python
def sort_priority3(numbers, group):
    found = False
    def helper(x):
        nonlocal found
        if x in group:
            found = True
            return (0, x)
        return (1, x)
    numbers.sort(key=helper)
    return found
```

nonlocal문은 closure에서 데이터를 다른 scope에 할당하는 시점을 알아보기 쉽게 해준다. nonlocal문은 변수 할당이 모듈 scope에 직접 들어가게 하는 global문을 보완한다. 하지만 전역 변수의 anti-pattern과 마찬가지로 간단한 함수 이외에는 nonlocal을 사용하지 않도록 주의해야 한다. nonlocal의 부작용은 알아내기 상당히 어렵고 특히 nonlocal문과 관련 변수에 대한 할당이 멀리 떨어진 긴 함수에서는 이해하기가 더욱 어렵다. 

nonlocal을 사용할 때 복잡해지기 시작하면 헬퍼 클래스로 상태를 감싸는 방법을 이용하는 게 낫다. (23에서 `__call__` 특별 메서드를 자세히 설명한다)

```python
class Sorter(object):
    def __init__(self, group):
        self.group = group
        self.found = False
    def __call__(self, x_):
        if x in self.group:
            self.found = True
            return (0, x)
        return (1, x)
        
sorter = Sorter(group)
numbers.sort(key=sorter)
assert sorter.found is True
```

* Closure 함수는 자신이 정의된 scope 중 어디에 있는 변수도 참조할 수 있다.
* 기본적으로 closure에서 변수를 할당하면 바깥쪽 scope에는 영향을 미치지 않는다.
* 파이썬3에서는 nonlocal문을 사용하여 closure를 감싸고 있는 scope 변수를 수정할 수 있음을 알린다.
* 간단한 함수 이외에는 nonlocal 문을 사용하지 말자.

### 16. List를 반환하는 대신 Generator를 고려하자

일련의 결과를 생성하고자 할 때 가장 간단한 방법은 아이템의 리스트를 반환하는 것이다. 예를 들어, 문자열에 있는 모든 단어의 인덱스를 출력하고 싶다고 하자.

```python
def index_words(text):
    result = []
    if text:
        result.append(0)
    for index, letter in enumerate(text):
        if letter == ' ':
            result.append(index + 1)
    return result
    
address = 'Four score and seven years ago...'
result = index_words(address)
print(result[:3])
>>>
[0, 5, 11]
```

샘플 입력이 적을 때는 함수가 기대한 대로 동작한다. 하지만, 위와 같은 append 기반의 함수는 다음과 같은 문제가 있다.

코드가 약간 복잡하고 깔끔하지 않다는 점이다. 새로운 결과가 나올 때마다 append 메서드를 호출해야 한다. 메서드 호출(result.append)가 많아서 리스트에 추가하는 값(index+1)이 덜 중요해 보인다. 결과 리스트를 생성하는 데 한 줄이 필요하고, 그 값을 반환하는 데도 한 줄이 필요하다. 함수 몸체에 문자가 130개 가량(공백 제외) 있지만 그중에서 중요한 문자는 약75개다.

Generator를 사용해서 더 좋은 함수를 작성해보자. Generator는 yield 표현식을 사용하는 함수로, 호출되면 실제로 실행하지 않고 바로 iterator를 반환한다. 내장 함수 next를 호출할 때마다 iterator는 generator가 다음 yield 표현식으로 진행하게 한다. Generator에서 yield에 전달한 값을 iterator가 호출하는 쪽에 반환한다.

```python
def index_words_iter(text):
    if text:
        yield 0
    for index, letter in enumerate(text):
        if letter == ' ':
            yield index + 1

result = list(index_words_iter(address))
```

결과 리스트와 연동하는 부분이 모두 사라져서 훨씬 이해하기 쉽다. 결과는 리스트가 아닌 yield 표현식으로 전달된다. Generator 호출로 반환되는 iterator를 내장 함수 list에 전달하면 손쉽게 리스트로 변환할 수 있다 (참고 9).

index_words의 두 번째 문제는 반환하기 전에 모든 결과를 리스트에 저장해야 한다는 점이다. 입력이 매우 많다면 프로그램 실행 중에 메모리가 고갈되어 동작을 멈추는 원인이 된다. 반면에 generator로 작성한 버전은 다양한 길이의 입력에도 쉽게 이용할 수 있다.

다음은 파일에서 입력을 한 번에 한 줄씩 읽어서 한 번에 한 단어씩 출력을 내어주는 geneartor이다. 이 함수가 동작할 때 사용하는 메모리는 입력 한 줄의 최대 길이까지다.

```python
def index_file(handle):
   offset = 0 
   for line in handle:
       if line:
           yield offset
       for letter in line:
           offset += 1
           if letter == ' ':
               yield offset

with open('/tmp/address.txt', 'r') as f:
    it = index_file(f)
    results = islice(it, 0, 3)
    print(list(results))
>>>
[0, 5, 11]
```

이와 같은 generator를 정의할 때 알아둬야 할 사항은 반환되는 iterator에 상태가 있고 재사용할 수 없다는 사실을 호출하는 쪽에서 알아야 한다는 점이다 (참고 17).

* Generator를 사용하는 방법이 누적된 결과의 리스트를 반환하는 방법보다 이해하기에 명확하다.
* Generator에서 반환한 iterator는 generator 함수의 본문에 있는 yield 표현식에 전달된 값들의 집합이다.
* Generator는 모든 입력과 출력을 메모리에 저장하지 않으므로 입력값의 양을 알기 어려울 때도 연속된 출력을 만들 수 있다.

### 17. 인수를 순회할 때는 방어적으로 하자

파라미터로 객체의 리스트를 받는 함수에서 리스트를 여러 번 순회해야 할 때가 종종 있다. 다음과 같이 평균을 내야하는 정규화 함수가 있다고 하자.

```python
def normalize(numbers):
	total = sum(numbers)
	result = []
	for value in numbers:
		percent = 100 * value / total
		result.append(percent)
	return result
	
visits = [15, 35, 80]
percentages = normalize(visits)
print(percentages)
>>>
[11.53846..., 26.9230..., 61.5384...]
```

굉장히 큰 visits 리스트를 계산하기 위해서 generator를 사용할 수 있다.

```python
def read_visits(data_path):
	with open(data_path) as f:
		for line in f:
			yield int(line)
			
it = read_visits('/tmp/my_numbers.txt')
percentages = normalize(it)
print(percentages)
>>>
[]
```

문제는 generator의 반환 값에 normalize를 호출하면 아무 결과도 생성되지 않는다. iterator가 결과를 한 번만 생성하기 때문이다. 이미 StopIteration 예외를 일으킨 iterator나 generator를 순회하면 어떤 결과도 얻을 수 없다. 

입력 iterator를 명시적으로 소진하고 전체 콘텐츠의 복사본을 리스트에 저장해야 한다. 이제 함수가 generator의 반환 값에도 올바르게 동작한다.

```python
def normalize_copy(numbers):
	numbers = list(numbers)	# iterator를 복사함
	total = sum(numbers)
	result = []
	for value in numbers:
		percent = 100 * value / total
		result.append(percent)
	return result
```

이 방법의 문제는 입력받은 iterator 콘텐츠의 복사본이 클 때 프로그램의 메모리가 고갈되어 동작이 멈출 수도 있다는 점이다. 이 문제를 피하기 위해 호출될 때마다 새 iterator를 반환하는 함수를 받게 만드는 것이다. normalize_func을 사용하려면 generator를 호출해서 매번 새 iterator를 생성하는 lambda 표현식을 넘겨주면 된다.

```python
def normalize_func(get_iter):
	total = sum(get_iter())	# 새 iterator
	result = []
	for value in get_iter():	# 새 iterator
		percent = 100 * value / total
		result.append(percent)
	return result
	
percentages = normalize_func(lambda: read_visits(path))
```

코드가 잘 동작하긴 하지만, 이렇게 lambda 함수를 넘겨주는 방법은 세련되지 못하다. 같은 결과를 얻는 더 좋은 방법은 iterator protocol을 구현한 새 컨테이너 클래스를 제공하는 것이다. iterator protocol은 파이썬의 for 루프와 관련 표현식이 컨테이너 타입의 콘텐츠를 탐색하는 방법이다.

파이썬은 for x in foo 같은 문장을 만나면 실제로 iter(foo)를 호출한다. 그러면 내장 함수 iter는 특별한 메서드인 `foo.__iter__`를 호출한다. `__iter__` 메서드는 (`__next__`라는 특별한 메서드를 구현하는) iterator 객체를 반환해야 한다. 마지막으로 for 루프는 iterator를 모두 소진할 때까지 (그래서 StopIteration 예외가 발생할 때까지) iterator 객체에 내장 함수 next를 계속 호출한다.

복잡해 보이지만 사실 클래스의 `__iter__` 메서드를 generator로 구현하면 이렇게 동작하게 만들 수 있다. 다음은 데이터를 담은 파일을 읽는 iterable 컨테이너 클래스이다. 새로 정의한 컨테이너 타입은 원래의 함수에 수정을 가하지 않고 넘겨도 제대로 동작한다.

```python
class ReadVisits(object):
  def __init__(self, data_path):
		self.data_path = data_path
		
	def __iter__(self):
		with open(self.data_path) as f:
			for line in f:
				yield int(line)

visits = ReadVisits(path)
percentages = normalize(visits)
print(percentages)
>>>
[11.53846..., 26.9230..., 61.5384...]
```

이 코드가 동작하는 이유는 normalize의 sum 메서드가 새 iterator 객체를 할당하려고 `ReadVisits.__iter__`를 호출하기 때문이다. 숫자를 정규화하는 for 루프도 두 번째 iterator 객체를 할당할 때 `__iter__`를 호출한다. 두 iterator는 독립적으로 동작하므로 각각의 순회 과정에 모든 입력 데이터 값을 얻을 수 있다. 이 방법의 유일한 단점은 입력 데이터를 여러 번 읽는다는 점이다.

한 가지 더 해야할 일은 normalize의 파라미터가 단순한 iterator가 아님을 보장하는 함수를 작성해야 한다. Protocol에 따르면 내장 함수 iter에 iterator를 넘기면 iterator 자체가 반환된다. 반면에 iter에 컨테이너 타입을 넘기면 매번 새 iterator 객체가 반환된다. 따라서, iterator면 TypeError를 일으켜 거부하게 만들면 된다.

```python
def normalize_defensive(numbers):
	if iter(numbers) is iter(numbers):	# iterator 거부!
		raise TypeError('Must supply a container')
	total = sum(numbers)
	result = []
	for value in numbers:
		percent = 100 * value / total
		result.append(percent)
	return result
``` 

normalize_defensive는 normalize copy처럼 입력 iterator 전체를 복사하고 싶지 않지만, 입력 데이터를 여러 번 순회해야 할 때 사용하면 좋다. 이 함수는 list와 ReadVisits를 입력으로 받으면 입력이 컨테이너이므로 기대한 대로 동작한다. Iterator protocol을 따르는 어떤 컨테이너 타입에 대해서도 제대로 동작할 것이다.

```python
visits = [15, 35, 80]
normalize_defensive(visits)	# 오류 없음
visits = ReadVisits(path)
normalize_defensive(visits)	# 오류 없음
```

함수는 입력이 iterable이어도 컨테이너가 아니면 예외를 일으킨다

```python
it = iter(visits)
normalize_defensive(it)
>>>
TypeError: Must supply a container
```

* 입력 인수를 여러 번 순회하는 함수를 작성할 때 주의하자. 입력 인수가 iterator라면 이상하게 동작해서 값을 잃어버릴 수 있다.
* 파이썬의 iterator protocol은 컨테이너와 iterator가 내장 함수 iter, next와 for 루프 및 관련 표현식과 상호 작용하는 방법을 정의한다.
* __iter__ 메서드를 generator로 구현하면 자신만의 iterable 컨테이너 타입을 쉽게 정의할 수 있다.
* 어떤 값에 iter를 두 번 호출했을 때 같은 결과가 나오고 내장 함수 next로 전진시킬 수 있다면 그 값은 컨테이너가 아닌 iterator이다.

### 18. 가변 위치 인수로 깔끔하게 보이게 하자

선택적인 위치 인수(이런 파라미터 이름을 관례적으로 `*args`라고 해서 종종 'star args'라고도 한다)를 받게 만들면 함수 호출을 더 명확하게 하고, 보기에 방해가 되는 요소를 없앨 수 있다. 예를 들어 디버그 정보 몇 개를 로그로 남긴다고 하자. 만약, 인수의 개수가 고정되어 있다면 메시지와 값 리스트를 받는 함수가 필요할 것이다. (리스트 자료 구조로 가변의 인자를 받을 수 있도록 해준다.)

```python
def log(message, values):
	if not values:
		print(messague)
	else:
		values_str = ', '.join(str(x) for x in values)
		print('%s: %s' % (message, values_str))

log('My numbers are', [1, 2])
log('Hi there', [])
>>>
My numbers are: 1, 2
Hi there
```

로그로 남길 값이 없을 때 빈 리스트를 넘겨야 한다는 건 불편하고 성가신 일이다. 두 번째 인수를 아예 남겨둔다면 더 좋을 것이다. 파이썬에서는 * 기호를 마지막 위치 파라미터 이름 앞에 붙이면 된다. 로그 메시지(log 함수의 message 인수)를 의미하는 첫 번째 파라미터는 필수지만, 다음에 나오는 위치 인수는 몇 개든 선택적이다. 따라서, 함수 본문은 수정할 필요가 없고 호출하는 쪽만 수정해주면 된다.

```python
def log(message, *values):	# 유일하게 다른 부분
	if not values:
		print(messague)
	else:
		values_str = ', '.join(str(x) for x in values)
		print('%s: %s' % (message, values_str))

log('My numbers are', [1, 2])
log('Hi there')	# 훨씬 나음
>>>
My numbers are: 1, 2
Hi there
```

리스트를 가변 인수 log 함수를 호출하는 데 사용하고 싶다면 다음과 같이 * 연산자를 붙이면 된다. 그러면 파이썬은 시퀀스에 들어 있는 아이템들을 위치 인수로 전달한다.

```python
favorites = [7, 33, 99]
log('Favorite colors', *favorites)
>>>
Favorite colors: 7, 33, 99
```

가변 개수의 위치 인수를 받는 방법에는 두 가지 문제가 있다.

첫 번째 문제는 가변 인수가 함수에 전달되기에 앞서 항상 tuple로 변환된다는 점이다. 이는 함수를 호출하는 쪽에서 generator에 * 연산자를 쓰면 generator가 모두 소진될 때까지 순회됨을 의미한다. 즉, 변환된 tuple은 generator로부터 생성된 모든 값을 담으므로 메모리를 많이 차지해 결국 프로그램을 망가지게 할 수 있다.

```python
def my_generator():
	for i in range(10):
		yield if
		
def my_func(*args):
	print(args)
	
it = my_generator()
my_func(*it)

>>>
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
```

`*args`를 받는 함수는 인수 리스트에 있는 입력의 수가 적당히 적다는 사실을 아는 상황에서는 좋은 방법이다. 이런 함수는 많은 리터럴이나 변수 이름을 한꺼번에 넘기는 함수 호출에 이상적이다. 주로 개발자들을 편하게 하고 코드의 가독성을 높이는 데 사용한다. (즉, `*args`를 쓴다면 입력의 수가 적당히 적다는 뜻이 내포되어 있다)

두 번째 문제는 추후에 호출 코드를 모두 변경하지 않고서는 새 위치 인수를 추가할 수 없다는 점이다. 인수 리스트의 앞쪽에 위치 인수를 추가하면 기존의 호출 코드가 수정 없이는 이상하게 동작한다.

```python
def log(sequence, message, *values):
	if not values:
		print('%s: %s' % (sequence, message))
	else:
		values_str = ', '.join(str(x) for x in values)
		print('%s: %s:' % (sequence, message, values_str))
		
log(1, 'Favorites', 7, 33)		# 새로운 용법은 OK
log('Favorite numbers', 7, 33)	# 오래된 용법은 제대로 동작하지 않음
>>>
1: Favorites: 7, 33
Favorite numbers: 7: 33
```

위 코드의 문제는 두 번째 호출이 sequence 인수를 받지 못해서 7을 message 파라미터로 사용한다는 점이다. (함수 정의는 새로운 버전인데 깜빡하고 예전 버전의 함수 호출을 사용할 수도 있으니...) 이런 버그는 코드에서 예외를 일으키지 않고 계속 실행되므로 발견하기가 극히 어렵다. 이런 문제가 생길 가능성을 완전히 없애려면 `*args`를 받는 함수를 확장할 때 키워드 전용 인수를 사용해야 한다 (ref.21).

* def 문에서 `*args`를 사용하면 함수에서 가변 개수의 위치 인수를 받을 수 있다
* `*`연산자를 사용하면 시퀀스에 들어 있는 아이템을 함수의 위치 인수로 사용할 수 있다
* Generator와 `*`연산자를 함께 사용하면 프로그램이 메모리 부족으로 망가질 수도 있다
* `*args`를 받는 함수에 새 위치 파라미터를 추가하면 정말 찾기 어려운 버그가 생길 수도 있다.

### 19. 키워드 인수로 선택적인 동작을 제공하자

대부분의 다른 프로그래밍 언어와 마찬가지로 파이썬에서도 함수를 호출할 때 인수를 위치로 전달할 수 있다. 즉, 파이썬 함수의 위치 인수를 모두 키워드로 전달할 수 있다. 키워드와 위치 인수를 섞어서 사용할 수도 있다.

```python
def remainder(number, divisor):
    return number % divisor
	
remainder(20, 7)
remainder(20, divisor=7)
remainder(number=20, divisor=7)
remainder(divisor=7, number=20)

remainder(number=20, 7) # 오류: 위치 인수는 키워드 인수 앞에 지정해야 한다
>>>
SyntaxError: non-keyword arg after keyword arg

remainder(20, number=7) # 오류: 각 인수는 한 번만 지정할 수 있다. (20과 7 모두 number로 인식)
>>>
TypeError: remainder() got multiple values for argument 'number'
```

키워드 인수의 유연성은 세 가지 중요한 이점이 있다.

키워드 인수의 첫 번째 이점은 코드를 처음 보는 사람이 함수 호출을 더 명확하게 이해할 수 있다는 점이다. 괄호 속 number와 divisor를 통해서 각각의 목적으로 어떤 파라미터를 사용했는지 곧바로 명확하게 알 수 있다.

키워드 인수의 두 번째 이점은 함수를 정의할 때 기본값을 설정할 수 있다는 점이다. 덕분에 함수에서 대부분은 기본값을 사용하지만 필요할 때 부가 기능을 제공할 수 있다. (단, 기본값이 복잡할 때는 다루기 까다롭다.ref.20) 이렇게 하면 코드가 깔끔해진다.

```python
def flow_rate(weight_diff, time_diff, period=1):   # period는 선택적인 인수이다.
    return (weight_diff / time_diff ) * period
	
flow_per_second = flow_rate(weight_diff, time_diff)
flow_per_hour = flow_rate(weight_diff, time_diff, period=3600)
```

키워드 인수의 세 번째 이점은 기존의 호출 코드와 호환성을 유지하면서도 함수의 파라미터를 확장할 수 있는 강력한 수단이 된다는 점이다. 이 방법을 쓰면 코드를 많이 수정하지 않고서도 추가적인 기능을 제공할 수 있고, 버그가 생길 가능성을 줄일 수 있다.

예를 들어, 킬로그램 단위는 물론 다른 무게 단위로도 유속을 계산하려고 앞의 flow_rate 함수를 확장한다고 하자. 원하는 측정 단위의 변환 비율을 새로운 선택 파라미터로 추가하여 확장하면 된다.

```python
def flow_rate(weight_diff, time_diff, period=1, units_per_kg=1):  
    return ((weight_diff / units_per_kg ) / time_diff ) * period

pounds_per_hour = flow_rate(weight_diff, time_diff, period=3600, units_per_kg=2.2)
pounds_per_hour = flow_rate(weight_diff, time_diff, 3600, 2.2) 
```

선택적인 인수로 위치로 넘기면 3600과 2.2값에 대응하는 인수가 무엇인지 명확하지 않아 혼동을 일으킬 수 있다. 가장 좋은 방법은 항상 키워드 이름으로 선택적인 인수를 지정하고 위치 인수로는 아예 넘기지 않는 것이다.

[Note] 이런 선택적인 키워드 인수를 사용하면 `*args`를 인수로 받는 함수에서 하위 호환성을 지키기 어렵다(ref.18). 더 좋은 방법은 키워드 전용 인수(ref.21)를 사용하는 것이다.

* 함수의 인수를 위치나 키워드로 지정할 수 있다.
* 위치 인수만으로는 이해하기 어려울 때 키워드 인수를 쓰면 각 인수를 사용하는 목적이 명확해진다.
* 키워드 인수에 기본값을 지정하면 함수에 새 동작을 쉽게 추가할 수 있다. 특히, 함수를 호출하는 기존 코드가 있을 때 사용하면 좋다.
* 선택적인 키워드 인수는 항상 위치가 아닌 키워드로 넘겨야 한다.

### 20. 동적 기본 인수를 지정하려면 None과 docstring을 사용하자

키워드 인수의 기본값으로 비정적(non-static) 타입을 사용해야 할 때도 있다. 예를 들어 이벤트 발생 시각까지 포함해 로깅 메시지를 출력한다고 하자. 기본적인 경우는 다음과 같이 함수를 호출한 시각을 메시지에 포함하려고 한다. 함수가 호출될 때마다 기본 인수를 평가한다고 가정하고 다음과 같이 처리하려 할 것이다.

```python
def log(message, when=datetime.now()):
    print('%s: %s' % (when, message))
	
log('Hi there!')
sleep(0.1)
log('Hi again!')

>>>
2014-11-15 21:10:10.371432: Hi there!
2014-11-15 21:10:10.371432: Hi again!
```

datetime.now 함수를 정의할 때 딱 한 번만 실행되므로 타임스탬프가 동일하게 출력된다.

파이썬에서 결과가 기대한 대로 나오게 하려면 기본값을 None으로 설정하고 docstring(문서화 문자열)으로 실제 동작을 문서화하는 게 관례다(ref.49). 코드에서 인수 값으로 None이 나타나면 알맞은 기본값을 할당하면 된다.

```python
def log(message, when=None):
    """Log a message with a timestamp.
	
	Args:
	    message: Message to print.
		when: datetime of when the message occurred.
		    Defaults to the present time.
	"""
	when = datetime.now() if when is None else when
	print('%s: %s' % (when, message))
	
log('Hi there!')
sleep(0.1)
log('Hi again!')

>>>
2014-11-15 21:10:10.472303: Hi there!
2014-11-15 21:10:10.573395: Hi again!
```

인수가 수정 가능(mutable)할 때, 기본 인수값으로 None을 사용하는 것이 중요하다. 예를 들어, json 데이터로 인코드된 값을 로드한다고 하자. 데이터 디코딩이 실패하면 기본값으로 빈 딕셔너리를 반환하려 한다.

```python
def decode(data, default={}):
    try:
        return json.loads(data)
	except ValueError:
		return default
		
foo = decode('bad data')
foo['stuff'] = 5
bar = decode('also bad')
bar['meep'] = 1
print('Foo:', foo)
print('Bar:', bar)

>>>
Foo: {'stuff': 5, 'meep': 1}
Bar: {'stuff': 5, 'meep': 1}
```

위의 코드는 datatime.now 예제와 같은 문제가 있다. 기본 인수값은 모듈이 로드될 때 딱 한 번만 할당받으므로, 기본값으로 설정한 딕셔너리를 모든 decode 호출에서 공유한다.

키워드 인수의 기본값을 None으로 설정하고 함수의 docstring에 동작을 문서화해서 이 문제를 해결한다.

```python
def decode(data, default=None):
	"""Load JSON data from a string.
	
	Args:
		data: JSON data to decode.
		default: Value to return if decoding fails.
			Defaults to an empty dictionary.
	"""
	if default is None:
		default = {}
    try:
        return json.loads(data)
	except ValueError:
		return default
		
foo = decode('bad data')
foo['stuff'] = 5
bar = decode('also bad')
bar['meep'] = 1
print('Foo:', foo)
print('Bar:', bar)

>>>
Foo: {'stuff': 5}
Bar: {'meep': 1}
```

* 기본 인수는 모듈 로드 시점에 함수 정의 과정에서 딱 한 번만 평가된다. 그래서 ({}나 []와 같은) 동적 값에는 이상하게 동작하는 원인이 된다.
* 값이 동적인 키워드 인수에는 기본값으로 None을 사용하고, 함수의 docstring에 실제 기본 동작을 문서화하자.

### 21. 키워드 전용 인수로 명료성을 강요하자

키워드로 인수를 넘기는 방법은 파이썬 함수의 강력한 기능이다(ref.19). 키워드 인수의 유연성 덕분에 쓰임새가 분명하게 코드를 작성할 수 있다.

예를 들어, 어떤 숫자를 다른 숫자로 나눈다고 해보자. 때로는 ZeroDivisionError 예외를 무시하고 무한대 값을 반환하고 싶을 수 있고, 어떨 때는 OverflowError 예외를 무시하고 0을 반환하고 싶을 수도 있다.

```python
def safe_division(number, divisor, 
				  ignore_overflow, ignore_zero_division):
    try:
		return number / divisor
	except OverflowError:
		if ignore_overflow:
			return 0
		else:
			raise
	except ZeroDivisionError:
		if ignore_zero_division:
			return float('inf')
		else:
			raise
			
result = safe_division(1, 10**500, True, False)
print(result) # 0.0
result = safe_division(1, 0, False, True)
print(result) # inf
```

문제는 예외 무시 동작을 제어하는 두 불 인수의 위치를 혼동하기 쉽다는 점이다. 이 때문에 찾기 어려운 버그가 쉽게 발생할 수 있다. 이런 코드의 가독성을 높이는 한 가지 방법은 키워드 인수를 사용하는 것이다. 그러면 호출하는 쪽에서 키워드 인수로 특정 연산에는 기본 동작을 덮어쓰고 무시할 플래그를 지정할 수 있다.

```python
def safe_division(number, divisor, 
				  ignore_overflow=False, 
				  ignore_zero_division=False):
	# ...
	
safe_division_b(1, 10**500, ignore_overflow=True)
safe_division_b(1, 0, ignore_zero_division=True)
```

또, 여기서 발생하는 문제는 이러한 키워드 인수가 선택적인 동작이라서 함수를 호출하는 쪽에 키워드 인수로 의도를 명확하게 드러내라고 '강요'할 방법이 없다는 점이다. safe_division_b라는 새 함수를 정의한다고 해도 여전히 위치 인수를 사용하는 이전 방식으로 호출할 수 있다.

```python
safe_division_b(1, 10**500, True, False)
```

복합한 함수를 작성할 때는 호출하는 쪽에서 의도를 명확히 드러내도록 강요하는 게 좋다. 파이썬 3에서는 키워드 전용 인수(keywork-only argument)로 함수를 정의해서 의도를 명확히 드러내도록 강요할 수 있다. 키워드 전용 인수는 키워드로만 넘길 뿐, 위치로는 절대 넘길 수 없다. 다음은 키워드 전용 인수로 safe_division을 다시 정의한 버전이다. 인수 리스트에 있는 * 기호는 위치 인수의 끝과 키워드 전용 인수의 시작을 가리킨다.

```python
def safe_division_c(number, divisor, *,
				  ignore_overflow=False, 
				  ignore_zero_division=False):
	# ...
	
safe_division_c(1, 10**500, True, False)

>>
TypeError: safe_division_c() takes 2 positional arguments but 4 were given

safe_division_c(1,0, ignore_zero_division=True)	# 문제 없음

try:
	safe_division_c(1, 0)
except ZeroDivisionError:
	pass	# 기대한 대로 동작
```

* 키워드 인수는 함수 호출의 의도를 더 명확하게 해준다
* 특히 bool 플래그를 여러 개 받는 함수처럼 혼동하기 쉬운 함수를 호출할 때 키워드 인수를 넘기게 하려면 키워드 전용 인수를 사용하자
* 파이썬3는 함수의 키워드 전용 인수 문법을 명시적으로 지원한다

<br>

<div id='3.'/>

## 3장. 클래스와 상속

파이썬은 상속, 다형성, 캡슐화 같은 객체 지향 언어의 모든 기능을 제공한다. 파이썬으로 작업을 처리하다 보면 새 클래스들을 작성하고 해당 클래스들이 인터페이스와 상속 관계를 통해 상호 작용하는 방법을 정의해야 하는 상황에 자주 접하게 된다. 

파이썬의 클래스와 상속을 이용하면 프로그램에서 의도한 동작을 객체들로 손쉽게 표현할 수 있다. 또한 프로그램의 기능을 점차 개선하고 확장할 수 있다. 아울러 요구 사항이 바뀌는 환경에서도 유연하게 대처할 수 있다. 클래스와 상속을 사용하는 방법을 잘 알아두면 유지보수가 용이한 코드를 작성할 수 있다.

### 22. 딕셔너리와 튜플보다는 헬퍼 클래스로 관리하자

파이썬에 내장되어 있는 딕셔너리 타입은 객체의 수명이 지속되는 동안 '동적인 내부 상태'(=예상하지 못한 식별자들을 관리해야 하는 상황)를 관리하는 용도로 사용하기 아주 좋다. 그러나, 딕셔너리는 정말 사용하기 쉬워서 과도하게 쓰다가 코드를 취약하게 만들 위험이 있다.

먼저, 이름을 모르는 학생 집단 성적을 기록해보자.

```python
class SimpleGradebook(object):
    def __init__(self):
        self._grades = {}
    def add_student(self, name):
        self._grades[name] = []
    def report_grade(self, name, score):
        self._grades[name].append(score)
    def average_grade(self, name):
        grades = self._grades[name]
	return sum(grades) / len(grades)
	
book = SimpleGradebook()
book.add_student('Issac Newton')
book.report_grade('Issac Newton', 90)
# ...
print(book.average_grade('Issac Newton'))
>>>
90.0
```

이제 SimpleGradebook 클래스를 확장해서 모든 성적을 한 곳에 저장하지 않고 과목별로 저장한다고 하자. 이 경우 `_grades` 딕셔너리를 변경해서 학생 이름(키)을 또 다른 딕셔너리(값)에 매핑하면 된다. 가장 안쪽에 있는 딕셔너리는 과목(키)을 성적(값)에 매핑한다.

```python
class BySubjectGradebook(object):
    # ...
    def report_grade(self, name, subject, grade):
        by_subject = self._grades[name]
	grade_list = by_subject.setdefault(subject, [])
	grade_list.append(grade)
    def average_grade(self, name):
        by_subject = self._grades[name]
	total, count = 0, 0
	for grades in by_subject.values():
	    total += sum(grades)
	    count += len(grades)
	return total / count
	
book = BySubjectGradebook()
book.add_student('Albert Einstein')
book.report_grade('Albert Einstein', 'Math', 75)
book.report_grade('Albert Einstein', 'Math', 65)
book.report_grade('Albert Einstein', 'Gym', 90)
book.report_grade('Albert Einstein', 'Gym', 95)
```

이제 요구사항이 좀 더 복잡해진다. 수업의 최종 성적에서 각 점수가 차지하는 비중을 매겨서 중간고사와 기말고사를 쪽지시험보다 더 중요하게 만들고자 한다. 가장 안쪽 딕셔너리를 변경해서 과목(키)을 성적(값)에 매핑하지 않고, 성적과 비중을 담은 튜플 (score, weight)에 매핑하면 된다.

```python
class WeightedGradebook(object):
    # ...
    def report_grade(self, name, subject, score, weight):
        by_subject = self._grades[name]
	grade_list = by_subject.setdefault(subject, [])
	grade_list.append((score, weight))   # 튜플로 저장
    def average_grade(self, name):
        by_subject = self._grades[name]
	score_sum, score_count = 0, 0
	for subject, scores in by_subject.items():
	    subject_avg, total_weight = 0, 0
	    for score, weight in scores:   # 루프 안에 루프가 생겨서 이해하기 어려워짐
	        # ...

book = WeightedGradebook()
# ...
book.report_grade('Albert Einstein', 'Math', 80, 0.10)   # 위치 인수에 있는 숫자들이 무엇을 의미하는지 명확하지 않음
```

이렇게 복잡해지면 딕셔너리와 튜플 대신에 클래스의 계층 구조를 사용할 때가 된 것이다. 처음엔 성적에 비중을 적용하게 될지 몰랐으니 복잡하게 헬퍼 클래스를 추가할 필요까지는 없다. 딕셔너리와 튜플 타입을 쓰면 내부 관리용으로 층층이 타입을 추가하는 게 쉽지만, 계층이 한 단계가 넘는 중첩은 피해야 한다. 즉, 딕셔너리를 담은 딕셔너리는 쓰지 말아야 한다. 여러 계층으로 중첩하면 다른 프로그래머들이 코드를 이해하기 어려워지고 유지보수의 악몽에 빠지게 된다.

관리가 복잡하다고 느껴진다면 클래스로 옮겨야 한다. 그러면 잘 캡슐화된 데이터를 정의할 수 있는 인터페이스를 제공할 수 있고, 인터페이스와 실제 구현 사이에 추상화 계층을 만들 수 있다. 그런데 일반 튜플의 문제점은 위치에 의존한다는 점이다.

** 클래스 리팩토링 **

의존 관계에서 가장 아래에 있는 성적부터 클래스로 옮겨보자. 사실, 이렇게 간단한 정보를 담기에 클래스는 너무 무거워 보인다. 성적은 변하지 않으니 튜플을 사용하는 게 더 적절해 보인다. 

```python
grades = []
grades.append((95, 0.45, 'Great job'))
# ...
total = sum(score * weight for score, weight, _ in grades)
total_weight = sum(weight for _, weight, _ in grades)
average_grade = total / total_weight
```
튜플을점점 더 길게 확장하는 패턴은 딕셔너리의 계층을 더 깊에 두는 방식과 비슷하다. 튜플의 아이템이 두 개를 넘어가면 다른 방법을 고려해야 한다.

collection 모듈의 namedtuple 타입이 정확히 이런 요구에 부합한다. namedtuple을 이용하면 작은 불변 데이터 클래스(immutable data class)를 쉽게 정의할 수 있다. 

```python
import collections
Grade = collections.namedtuple('Grade', ('score', 'weight'))
```

불변 데이터 클래스는 위치 인수나 키워드 인수로 생성할 수 있다. 필드는 이름이 붙은 속성으로 접근할 수 있다. 이름이 붙은 속성이 있으면 나중에 요구 사항이 변해서 단순 데이터 컨테이너에 동작을 추가해야 할 때 namedtuple에서 직접 작성한 클래스로 쉽게 바꿀 수 있다.

namedtuple의 제약 <br>
namedtuple이 여러 상황에서 유용하지만 단점을 만들어내는 상황을 이해해야 한다.

* namedtuple로 만들 클래스에 기본 인수 값을 설정할 수 없기 때문에 데이터에 선택적인 속성이 많으면 다루기 힘들어진다. 속성을 사용할 때는 클래스를 직접 정의하는 게 나을 수 있다.
* namedtuple 인스턴스의 속성 값을 여전히 숫자로 된 인덱스와 순회 방법으로 접근할 수 있다. 특히 외부 API로 노출한 경우에는 의도와 다르게 사용되어 나중에 실제 클래스로 바꾸기 더 어려울 수 있다. namedtuple 인스턴스를 사용하는 방식을 모두 제어할 수 없다면 클래스를 직접 정의하는 게 낫다.

이제 성적 이외에 나머지 것들을 클래스로 작성해보자.

```python
class Subject(object):
    """ 단일 과목을 표현 """
    def __init__(self):
        self._grades = []
    def report_grade(self, score, weight):
        self._grades.append(Grade(score, weight))   # namedtuple 사용
    def average_grade(self):
        total, total_weight = 0, 0
	for grade in self._grades:
	    total += grade.score * grade.weight
	    total_weight += grade.weight
	return total / total_weight

class Student(object):
    """ 한 학생이 공부한 과목을 표현 """
    def __init__(self):
        self._subjects = {}
    def subject(self, name):
        if name not in self._subjects:
	    self._subjects[name] = Subject()
	return self._subjects[name]
    def average_grade(self):
        total, count = 0, 0
	for subject in self._subjects.values():
	    totoal += subject.average_grade()
	    count += 1
	return total / count
	
class Gradebook(object):
    """ 학생의 이름을 키로 사용해 동적으로 모든 학생을 담을 컨테이너 """
    def __init__(self):
        self._students = {}
    def student(self, name):
        if name not in self._students:
	    self._students[name] = Student()
	return self._students[name]

book = Gradebook()
albert = book.student('Albert Einstein')
math = albert.subject('Math')
math.report_grade(80, 0.10)
# ...
print(albert.average_grade())
>>>
81.5
```

위의 세 클래스의 코드 줄 수는 이전에 구현한 코드의 두 배에 가깝다. 하지만, 이 코드가 훨씬 이해하기 쉽다. 이 클래스를 사용하는 예제도 더 명확하고 확장하기 쉽다. 필요하면 이전 형태의 API 스타일로 작성한 코드를 새로 만든 객체 계층 스타일로 바꿔주는 하위 호환용 메서드를 작성해도 된다.

* 다른 딕셔너리나 긴 튜플을 값으로 담은 딕셔너리를 생성하지 말자
* 정식 클래스의 유연성이 필요 없다면 가벼운 불변 데이터 컨테이너에는 namedtuple을 사용하자
* 내부 상태를 관리하는 딕셔너리가 복잡해지면 여러 헬퍼 클래스를 사용하는 방식으로 관리 코드를 바꾸자


### 23. 인터페이스가 간단하면 클래스 대신 함수를 받자



<br>

<div id='5.'/>

## 5장. 병행성과 병렬성

* 병행성(concurrency)는 컴퓨터가 여러 일을 마치 동시에 하듯이 수행하는 것이다. 
   * 예를 들어, CPU 코어가 하나인 컴퓨터에서 운영체제는 단일 프로세서에서 실행하는 프로그램을 빠르게 변경한다. 
   * 이 방법으로 프로그램을 교대로 실행하여 프로그램들이 동시에 실행하는 것처럼 보이게 한다.
* 병렬성(parallelism)은 실제로 여러 작업을 동시에 실행하는 것이다. 
   * CPU 코어가 여러 개인 컴퓨터는 여러 프로그램을 동시에 실행할 수 있다. 
   * 각 CPU 코어가 각기 다른 프로그램 명령어를 실행하여 각 프로그램이 같은 순간에 실행하게 해준다.
* 단일 프로그램 안에서 병행성이라는 도구를 이용하면 특정 유형의 문제를 더욱 쉽게 해결할 수 있다. 
* 병행 프로그램은 별개의 여러 실행 경로를 동시에 독립적으로 실행하는 것처럼 진행하게 해준다. 
* 병행성과 병렬성 사이의 가장 큰 차이점은 속도 향상이다. 
   * 한 프로그램에서 서로 다른 두 실행 경로를 병렬로 진행하면 전체 작업에 걸리는 시간이 절반으로 준다. 
   * 반면에 병행 프로그램은 수천가지 실행 경로를 병렬로 수행하는 것처럼 보이게 해주지만 전체 작업 속도는 향상되지 않는다.
* 파이썬을 쓰면 병행 프로그램을 쉽게 작성할 수 있다. 
* 시스템 호출, 서브프로세스, C 확장을 이용한 병렬 작업에도 파이썬을 쓸 수 있다. 
* 그러나 병행 파이썬 코드를 실제 병렬로 실행하게 만드는 건 정말 어렵다. (파이썬 특징이기도 하다)

<br>

### 36. 자식 프로세스를 관리하려면 subprocess를 사용하자

...

<br>

### 37. 스레드는 블로킹 I/O용으로 사용하고 병렬화용으로는 사용하지 말자

* 파이썬의 표준 구현을 CPython이라고 한다. CPython은 파이썬 프로그램을 두 단계로 실행한다. 
   * 먼저 소스 텍스트를 바이트코드(bytecode)로 파싱하고 컴파일한다. 
   * 그런 다음 스택 기반 인터프리터로 바이트코드를 실행한다. 
* 바이트코드 인터프리터는 파이썬 프로그램이 실행되는 동안 지속되고, 일관성 있는 상태를 유지한다. 
* 파이썬은 전역 인터프리터 잠금(GIL, Global Interpreter Lock)이라는 메커니즘으로 일관성을 유지한다. 
* 본질적으로 GIL은 상호 배제 잠금(mutex; 뮤텍스)이며 CPython이 선점형 멀티스레딩의 영향을 받지 않게 막아준다. 
   * 선점형 멀티스레딩(preemptive multithreading)은 한 스레드가 다른 스레드를 인터럽트(차단)해서 프로그램의 제어를 얻는 것을 말한다. 
   * 그런데 이 인터럽트가 예상치 못한 시간에 일어나면 인터프리터 상태가 망가진다. 
   * GIL은 이런 인터럽트를 막아주며 모든 바이트코드 명령어가 CPython 구현과 C 확장 모듈에서 올바르게 동작함을 보장한다.
* GIL이 중요한 부작용을 가지고 있다. C++이나 자바 같은 언어로 작성한 프로그램에서 여러 스레드를 실행한다는 건 프로그램이 동시에 여러 CPU 코어를 사용함을 의미한다. 파이썬도 멀티스레드를 지원하지만, GIL은 한 번에 한 스레드만 실행하게 한다. 
   * 다시 말해 스레드가 병렬 연산을 해야 하거나 파이썬 프로그램의 속도를 높여야 하는 상황이라면 실망하게 될 것이다.
* 예를 들어 파이썬으로 연산 집약적인 작업을 한다고 해보자. 여기서는 단순 숫자 인수 분해 알고리즘을 사용한다.

```python
def factorize(number):
    for i in range(1, number + 1):
	    if number % i == 0:
		    yield i
			
# 순서대로 숫자들을 인수 분해하면 시간이 꽤 오래 걸린다.
numbers = [2139079, 1214759, 1516637, 1852285]
start = time()
for number in numbers:
    list(factorize(number))
end = time()
print('Took %.3f seconds' % (end - start))
```
```
>>>
Took 1.040 seconds
``` 

다른 언어에서는 당연히 이런 연산에 멀티스레드를 이용한다. 멀티스레드를 이용하면 컴퓨터의 모든 CPU를 최대한 활용할 수 있다. 여기서 위의 연산을 파이썬으로 정의한다.

```python
from threading import Thread

class FactorizeThread(Thread):
    def __init__(self, number):
	    super().__init__()
		self.number = number
	
	def run(self):
	    self.factors = list(factorize(self.number))
		
# 다음으로 각 숫자를 인수 분해할 스레드를 병렬로 시작한다.
start = time()
threads = []
for number in numbers:
    thread = FactorizeThread(number)
	thread.start()
	threads.append(thread)
	
# 마지막으로 스레드들이 모두 완료하기를 기다린다.
for thread in threads:
    thread.join()
end = time()
print('Took %.3f seconds' % (end - start))
```
```
>>>
Took 1.061 seconds
```

놀라운 사실은 순서대로 인수 분해할 때보다 많은 시간이 걸렸다는 점이다. 이 코드를 듀얼코어 머신에서 실행한다면 2배 정보의 속도 향상을 기대했을 것이다. 하지만 멀티코어임에도 이 스레드들의 성능이 더 나쁘게 나왔다. 이로부터 GIL이 표준 CPython 인터프리터에서 실행하는 프로그램에 미치는 영향을 알 수 있다.

CPython이 멀티코어를 활용하게 하는 방법은 여러 가지가 있지만, 표준 Thread 클래스에는 동작하지 않으므로(Better way 41) 노력이 필요하다. 이런 제약을 알게 되면 파이썬이 스레드를 왜 지원하는지 의문이 들지도 모른다. 여기에는 두 가지 좋은 이유가 있다.

첫 번째 이유는 멀티스레드를 이용하면 프로그램이 동시에 여러 작업을 하는 것처럼 만들기가 용이하다. 동시에 동작하는 태스크를 관리하는 코드를 직접 구현하기는 어렵다(Better way 40). 스레드를 이용하면 함수를 마치 병렬로 실행하는 것처럼 해주는 일을 파이썬에 맡길 수 있다. 비록 GIL 때문에 한 번에 한 스레드만 진행하지만, CPython은 파이썬 스레드가 어느 정도 공평하게 실행됨을 보장한다.

파이썬이 스레드를 지원하는 두 번째 이유는 특정 유형의 시스템 호출을 수행할 때 일어나는 블로킹 I/O를 다루기 위해서다. 시스템 호출(system call)은 파이썬 프로그램에서 외부 환경과 댓신 상호 작용하도록 컴퓨터 운영체제에 요청하는 방법이다. 블로킹 I/O로는 파일 읽기/쓰기, 네트워크와의 상호작용, 디스플레이 같은 장치와의 통신 등이 있다. 스레드는 운영체제가 이런 요청에 응답하는 데 드는 시간을 프로그램과 분리하므로 블로킹 I/O를 처리할 때 유용하다.

예를 들어, 원격 제어가 가능한 헬리콥터에 직렬 포트로 신호를 보내고 싶다고 하자. 예제에서는 이 작업을 느린 시스템 호출(select)에 위임한다. 이 함수는 동기식 직렬 포트를 사용할 때 일어나는 상황과 비슷하게 하려고 운영체제에 0.1초간 블록한 후 제어를 프로그램에 돌려달라고 요청한다.

```python
import select

def slow_systemcall():
    select.select([], [], 0.1)
	
# 이 시스템 호출을 연속해서 실행하면 시간이 선형으로 증가한다.
start = time()
for _ in range(5):
    slow_systemcall()
end = time()
print('Took %.3f seconds' % (end - start))
>>>
Took 0.503 seconds
```

문제는 slow_systemcall 함수가 실행되는 동안에는 프로그램이 다른 일을 할 수 없다는 점이다. 프로그램의 메인 스레드는 시스템 호출 select 때문에 실행이 막혀 있다. 실제로 벌어진다면 끔찍한 상황이다. 신호를 헬리콥터에 보내는 동안 헬리콥터의 다음 이동을 계산해야 한다. 그렇지 않으면 헬리콥터가 충돌할 것이다. 블로킹 I/O를 사용하면서 동시에 연산도 해야 한다면 시스템 호출을 스레드로 옮기는 방안을 고려해야 한다. 

다음 코드는 slow_systemcall 함수를 별도의 스레드에서 여러 번 호출하여 실행한다. 이렇게 하면 동시에 여러 직렬 포트(및 헬리콥터)와 통신할 수 있게 되고, 메인 스레드는 필요한 계산이 무엇이든 수행하도록 남겨둘 수 있다.

```python
start = time()
threads = []
for _ in range(5):
    thread = Thread(target=slow_systemcall)
    thread.start()
    threads.append(thread)

# 스레드가 시작하면 시스템 호출 스레드가 종료할 때까지 기다리기 전에 헬리콥터의 다음 이동을 계산한다.
def compute_helicopter_location(index):
    # ...
	
for i in range(5):
    compute_helicopter_location(i)
for thread in threads:
    thread.join()
end = time()
print('Took %.3f seconds' % (end - start))
>>>
Took 0.102 seconds
```

병렬 처리 시간은 직렬 처리 시간보다 5배나 짧아졌다. 이 예제는 시스템 호출이 GIL의 제약을 받지만 여러 파이썬 스레드를 모두 병렬로 실행할 수 있음을 보여준다. GIL은 파이썬 코드가 병렬로 실행하지 못하게 한다. 하지만 시스템 호출에서는 이런 부정적인 영향이 없다. 이는 파이썬 스레드가 시스템 호출을 만들기 전에 GIL을 풀고 시스템 호출의 작업이 끝나는 대로 GIL을 다시 얻기 때문이다. 

스레드 이외에도 내장 모듈 asyncio처럼 블로킹 I/O를 다루는 다양한 수단이 있고, 이런 대체 수단에는 중요한 이점이 있다. 하지만 이런 옵션을 선택하면 실행 모델에 맞춰서 코드를 재작성해야 하는 추가 작업이 필요하다(Better way 40). 스레드를 시용하는 방법은 프로그램의 수정을 최소화하면서도 블로킹 I/O를 병렬로 수행하는 가장 간단한 방법이다.

핵심 정리
* 파이썬 스레드는 전역 인터프리터 잠금(GIL, Global Interpreter Lock) 때문에 여러 CPU 코어에서 병렬로 바이트코드를 실행할 수 없다.
* GIL에도 불구하고 파이썬 스레드는 동시에 여러 작업을 하는 것처럼 보여주기 쉽게 해주므로 여전히 유용하다.
* 여러 시스템 호출을 병렬로 수행하려면 파이썬 스레드를 사용하자. 이렇게 하면 계산을 하면서도 블로킹 I/O를 수행할 수 있다.


### 38.



### 39.



### 40.



### 41. 진정한 병렬성을 실현하려면 concurrent.futures를 고려하자






<br>

<div id='6.'/>

## 6장. 내장 모듈

몇몇 표준 내장 패키지는 언어 사양의 일부이므로 파이썬의 특징과 밀접하게 관련있다. 이런 기본적인 내장 모듈은 복잡한 트로그램을 작성하거나 오류가 발생할 가능성이 큰 프로그램을 작성할 때 특히 중요하다.

### 42.

### 43.

### 44. 

### 45. 

### 46. 내장 알고리즘과 자료 구조를 사용하자

많은 데이터를 처리하는 파이썬 프로그램을 구현하다 보면 (파이썬 언어의 속도 때문이 아닌) 여러분이 작성한 코드의 알고리즘 복잡도 때문에 속도가 떨어지는 현상을 보게 된다. 최적의 알고리즘과 자료 구조를 사용할 필요가 있다.

파이썬 표준 라이브러리에는 필요한 만큼 많은 알고리즘과 자료 구조가 있다. 

#### Double-ended Queue

collection 모듈의 deque 클래스는 더블 엔디드 큐이다. Deque는 큐의 처음과 끝에서 아이템을 삽입하거나 삭제할 때 항상 일정한 시간이 걸리는 연산을 제공한다. 이는 선입선출(FIFO, Fist-In-First-Out) 큐를 만들 때 이상적이다.

```python
fifo = deque()
fifo.append(1)        # 생산자
x = fifo.popleft()    # 소비자
```

내장 타입 리스트(list)도 큐와 같이 순서가 있는 아이템 시퀀스를 담는다. 일정한 시간 내에 리스트의 끝에서 아이템을 삽입하거나 삭제할 수 있다. 하지만, 리스트의 시작 부분에서 아이템을 삽입하거나 삭제하는 연산에는 선형적 시간(linear time)(`*`횟수에 따라 시간도 늘어남)이 걸리므로 deque의 일정한 시간보다 훨씬 느리다.

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


### 47.

### 48. 




<br>

<div id='7.'/>

## 7장. 협력

* 파이썬은 명확한 인터페이스 경계로 잘 정의된 API를 만드는 데 도움을 주는 언어 기능을 갖췄다.
* 파이썬 커뮤니티는 시간이 지나도 코드의 유지보수성을 극대화할 수 있는 가장 좋은 방법을 확립했다.
* 서로 다른 환경으로 일하는 대규모 팀들이 협력할 수 있게 해주는 표준 도구들도 파이썬에 같이 실려온다.

### 49. 모든 함수, 클래스, 모듈에 docstring을 작성하자

* 파이썬에서 문서화는 언어의 동적 특성 때문에 극히 중요하다.
* 파이썬은 코드 블록에 문서를 첨부하는 기능을 기본으로 지원한다.
* 대부분의 다른 언어와는 다르게 프로그램 실행 중에 소스 코드에 첨부한 문서에 직접 접근할 수 있다.
* 다음은 함수 docstring.

```python
def palindrome(word):
    """Return True if the given word is a palindrome."""
    return word == word[::-1]

print(repr(palindrome.__doc__))
>>>
Return True if the given word is a palindrome.
``` 

* docstring은 함수, 클래스, 모듈에 붙일 수 있다. 이와 같은 연결은 파이썬 프로그램을 컴파일하고 실행하는 과정의 일부다.
* docstring과 __doc__ 속성을 지원하는 덕분에 다음 세 가지 효과를 얻는다.
   * 문서의 접근성 덕분에 대화식으로 개발하기가 더 쉽다. 
      * 파이썬 대화식 인터프리터(파이썬 쉘, 주피터 노트북)과 같은 도구
      * 내장 함수 help로 함수, 클래스, 모듈을 조사하여 문서를 볼 수 있다.
   * 문서를 정의하는 표준 방법이 있으면 텍스트를 더 쉽게 이해할 수 있는 포맷(HTML)으로 변환하는 도구를 쉽게 만들 수 있다.
      * 파이썬 커뮤니티의 Sphinx와 같은 훌륭한 문서 생성 도구가 있음.
      * 파이썬 프로젝트의 멋진 문서를 제공하는 Read the Docs와 같은 커뮤니티 펀딩 사이트도 있음.
   * 파이썬의 일급 클래스(first-class), 접근성, 잘 정리된 문서는 사람들이 문서를 더 많이 작성할 수 있도록 북돋아준다.
      * '좋은 코드'는 문서화가 잘 된 코드이다.
      * 대부분의 오픈 소스 파이썬 라이브러리가 잘 작성된 문서를 갖추고 있다.
* 이러한 훌륭한 문서화에 동참하려면 docstirng을 작성할 때 몇 가지 지침을 따라야 한다. -> PEP 257

#### 모듈 문서화

* 각 모듈에는 최상위(top-level) docstring이 있어야 한다. (소스 파일에서 첫 번째 문장에 있는 문자열)
* 최상위 docstring에는 큰따옴표 세 계(""")를 사용해야 한다.
* 이 docstring의 목적은 모듈과 모듈의 내용에 대한 소개다.
* 모듈 docstring 내용
   * 첫 번째 줄: 모듈의 목적을 기술하는 한 문장으로 구성
   * 그 이후의 문단
      * 모듈의 모든 사용자가 알아야 하는 모듈의 동작을 자세히 설명한 내용을 포함
      * 모듈 내의 중요 클래스나 함수를 강조
* 모듈이 명령줄 유틸리티라면 모듈 docstring이야 말로 명령줄에서 도구를 실행하는 방법을 보여주기에 적합한 곳이다.

```python
# words.py
# !/usr/bin/env python3
"""Library for testing words for various linguistic patterns.

Testing how words related to each other can be tricky sometimes!
THis module provides easy ways to determine when wods you've
found have special properties.

Available functions:
- palindrome: Determine if a word is a palindrome.
- check_anagram: Detmine if two words are anagrams.
...
```

#### 클래스 문서화

* 각 클래스에는 클래스 수준 docstring이 있어야 한다. (모듈 수준 docstring과 거의 같은 패턴)
* 클래스 docstring 내용
   * 클래스의 중요한 공개 속성과 메서드 강조
   * 서브클래스가 보호 속성, 슈퍼클래스의 메서드와 올바르게 상호 작용하는 방법을 안내 (Better way 27 참고)

```python
class Player(object):
    """Represents a player of the game.

    Subclasses may override the 'tick' method to provide
    custom animations for the player's movement depending
    on their power level, etc.

    Public attributes:
    - power: Unused power-ups (float between 0 and 1).
    - coins: Coins found during the level (integer).
    """

    # ...
```

#### 함수 문서화

* 각 공개 함수와 메서드에는 docstring이 있어야 한다.
* 이 docstring도 모듈이나 클래스와 같은 패턴을 따름.
  * 반환값 언급
  * 호출하는 쪽에서 함수 인터페이스의 일부로 처리해야 하는 예외도 설명

```python
def find_anagrams(word, dictionary):
    """Find all anagrams for a word.

    This function only runs as fast as the test for
    membership in the 'dictionary' container. It will
    be slow if the dictionary is a list and fast if
    it's a set.

    Args:
        word: String of the target word.
        dictionary: Container with all strings that
            are known to be actual words.

    Returns:
        List of anagrams that were found. Empty if
        none were found.
    """

    # ...
```

함수 docstring을 작성할 때 몇 가지 특별한 경우

* 함수가 인수는 받지 않고 간단한 값만 반환할 때는 한 줄짜리 설명으로 충분하다.
* 함수가 아무것도 반환하지 않으면 `return None` 보다는 반환값을 언급하지 않는 게 낮다.
* 함수가 일반적인 동작에서는 예외를 일으키지 않는다고 생각한다면 이에 대해서는 언급하지 말자.
* 함수가 받는 인수의 개수가 가변적(Better way 18)이거나 키워드 인수(Better way 19)를 사용한다면 문서의 인수 목록에 `*args`와 `**kwargs`를 사용해서 그 목적을 설명하자.
* 함수가 기본값이 있는 인수를 받는다면 해당 기본값을 설명해야 한다(Better way 20).
* 함수가 generator(Better way 16)라면 generator가 순환될 때 무엇을 넘겨주는지 설명해야 한다.
* 함수가 코루틴(Better way 40)이라면 코루틴이 무엇을 넘겨주는지, yield 표현식으로부터 무엇을 얻는지, 언제 순회가 끝나는지를 설명해야 한다.

Note

* 모듈의 docstring을 작성한 후에는 문서를 계속 업데이트하는 게 중요하다.
* 내장 모듈 doctest는 docstring에 포함된 사용 예제를 실행하기 쉽게 해줘서 여러분이 작성한 소스 코드와 문서가 시간이 지나면서 여러 버전으로 나뉘지 않게 해준다.

핵심 정리

* 모든 모듈, 클래스, 함수를 docstring으로 문서화하자. 코드를 업데이트할 때마다 관련 문서도 업데이트하자.
* 모듈: 모듈의 내용과 모든 사용자가 알아둬야 할 중요한 클래스와 함수를 설명한다.
* 클래스: class 문 다음의 docstring에서 클래스의 동작, 중요한 속성, 서브클래스의 동작을 설명한다.
* 함수와 메서드: def 문 다음의 docstring에서 모든 인수, 반환 값, 일어나는 예외, 다른 동작들을 문서화한다.

### 50.

...


<br>

<div id='8.'/>

## 8장. 제품화

* 작성한 파이썬 프로그램을 사용하려면 개발 환경에서 재품 환경으로 옮겨야 한다.
* 다양한 상황에서 신뢰할 만한 프로그램을 만드는 것은 올바르게 동작하는 프로그램을 만드는 것만큼이나 중요하다.
* 목표는 파이썬 프로그램을 제품화해서 프로그램을 사용하는 동안 문제가 없게 하는 것이다.
* 파이썬은 프로그램을 견고하게 해주는 내장 모듈을 갖추고 있고, 이 모듈은 디버깅, 최적화, 실행 시 프로그램의 품질과 성능을 극대화하는 테스트 기능을 제공한다.

### 54. 배포 환경을 구성하는 데는 모듈 스코프 코드를 고려하자

* 모든 프로그램에는 적어도 하나의 배포 환경(deployment environment)과 제품 환경(production environment)이 있다.
* 배포 환경은 프로그램을 실행하는 구성을 말한다.
* 프로그램을 작성하는 첫 번째 목적은 제품 환경에서 프로그램이 동작하도록 하고 결과를 얻게 하는 것이다.
* 프로그램을 작성하거나 수정하려면 개발에 사용 중인 컴퓨터에서 프로그램이 동작하게 해야 한다.
* 개발 환경(development environment)의 설정은 제품 환경과는 많이 다르다. (ex. 리눅스 워크스테이션으로 슈퍼컴퓨터용 프로그램을 작성)
* pyvenv(Better way 53) 같은 도구를 이용하면 모든 환경에 같은 파이썬 패키지가 설치되도록 보장하기가 쉽다.
* 문제는 제품 환경에서는 종종 개발 환경에서 재현하기 어려운 많은 외부 기능을 요구한다는 점이다.
   * 예를 들어, 프로그램이 웹 서버 컨테이너에서 실행되고 데이터베이스에 접근해야 한다고 하자. 
   * 프로그램을 수정할 때마다 서버 컨테이너를 실행해야 하고, 데이터베이스를 적절하게 설정해야 한다.
* 이런 문제를 해결하는 가장 좋은 방법은 시작할 때 프로그램의 일부를 override해서 배포 환경에 따라 서로 다른 기능을 제공하는 것이다.
   * 예를 들면, 서로 다른 __main__ 파일을 두 개 만든다. 하나는 제품용, 하나는 개발용으로 사용한다.

```python
# dev_main.py
TESTING = True
import db_connection
db = db_connection.Database()

# prod_main.py
TESTING = False
import db_connection
db = db_connection.Database()

# ---

# db_connection.py
import __main__

class TestingDatabase(object):
    # ...

class RealDatabase(object):
    # ...

if __main__.TESTING:
    Database = TestingDatabase
else:
    Database = RealDatabase
```

* 여기서 알아야 할 중요한 동작은 (함수나 메서드 내부가 아닌) 모듈 스코프에서 동작하는 코드는 일반 파이썬 코드라는 점이다.
* 모듈 수준에서 if 문을 이용하여 모듈이 이름을 정의하는 방법을 결정할 수 있다.
* 이 방법으로 모듈을 다양한 배포 환경에 알맞게 만들 수 있다.
* 또한, 데이터베이스 설정 등이 필요 없을 때 재현해야 하는 수고를 덜 수 있다.
* 목(mock)이나 가짜 구현을 주입하여 대화식 개발과 테스트를 요잉하게 할 수도 있다(Better way 56)
* Note
   * 배포환경이 복잡해지면 (TESTING 같은) 파이썬 상수에서 별도의 설정 파일로 옮기는 방안을 고려해야 한다.
   * 내장 모듈 configparser와 같은 도구를 이용하면 제품 설정을 코드와 분리해서 관리할 수 있으며, 운영팀과 협력할 때는 반드시 이렇게 분리해야 한다.
* 이 방법을 외부의 전제를 우회하는 목적 이외에도 사용할 수 있다.
   * 예를 들어, 프로그램이 호스트 플랫폼에 따라 다르게 동작해야 한다면 모듈의 최상위 구성 요소를 정의하기 전에 sys 모듈을 조사하면 된다.

```python
# db_connection.py
import sys

class Win32Database(object):
    # ...

class PosixDatabase(object):
    # ...

if sys.platform.startswith('win32'):
    Database = Win32Database
else:
    Database = PosixDatabase
```

* 이와 유사하게 os.environ에 들어 있는 환경 변수를 기반으로 모듈을 정의할 수 있다.

핵심 정리 

* 종종 프로그램을 여러 배포 환경에서 실행해야 하며, 각 환경마다 고유한 전제와 설정이 있다.
* 모듈 스코프에서 일반 파이썬 문장을 사용하여 모듈 컨텐츠를 다른 배포 환경에 맞출 수 있다.
* 모듈 컨텐츠는 sys와 os 모듈을 이용한 호스트 조사 내역 같은 외부 조건의 결과물이 될 수 있다.


### 55. 디버깅 출력용으로는 repr 문자열을 사용하자.

* 파이썬 프로그램을 디버깅할 때 print 함수(혹은 내장 logging 모듈을 이용한 출력)는 놀랍게도 많은 일을 한다.
* 보통 파이썬 내부를 일반 속성으로 쉽게 접근할 수 있다(Better way 27).
* print로 프로그램이 실행 중에 상태가 어떻게 변하는지 출력하여 어떤 부분이 제대로 동작하지 않는지 보기만 하면 된다.
* print 함수는 사람이 읽기 쉬운 문자열 버전으로 결과를 출력한다.
* 문제는 사람이 읽기 쉬운 문자열로는 값의 실제 타입이 무엇인지 명확하게 파악하기 어렵다는 점이다.

```python
print(5)
print('5')
```
```
>>>
5
5
```

* print로 프로그램을 디버깅한다면 위와 같은 종류의 차이는 문제가 된다.
* 디버깅 중에는 대부분 객체의 repr 버전을 보려고 한다.
* 내장 함수 repr은 객체의 출력 가능한 표현을 반환하며, 이 표현은 객체를 가장 명확하게 이해할 수 있는 문자열 표현이어야 한다.
* 내장 타입인 경우 repr이 반환하는 문자열은 올바른 파이썬 표현식이다.

```python
a = '\x07'
print(repr(a))
```
```
>>>
'\x07'
```

* repr이 반환한 값을 내장 함수 eval에 전달하면 원래의 파이썬 객체와 동일한 결과가 나와야 한다. (물론 실제로는 eval을 아주 조심해서 사용해야 한다.)

```python
b = eval(repr(a))
assert a == b
```

* print로 디버깅할 때는 타입의 차이가 명확히 드러나도록 값을 출력하기 위해, repr을 사용해야 한다.

```python
print(repr(5))
print(repr('5'))
```
```
>>>
5
'5'
```

* 위의 결과는 '%r' 포맷 문자열과 % 연산자로 출력한 결과와 같다.

```python
print('%r' % 5)
print('%r' % '5')
```
```
>>>
5
'5'
```

* 동적 파이썬 객체의 경우 기본으로 사람이 이해하기 쉬운 문자열 값이 repr값과 같다. 
* 이는 print에 동적 객체를 넘기면 제대로 동작하므로 명시적으로 repr을 호출하지 않아도 됨을 의미한다.
* 불행하게도 object 인스턴스에 대한 repr의 기본값은 특별히 도움이 되지 않는다. 예를 들어, 간단한 클래스를 정의하고 그 값을 출력해보자.

```python
class OpaqueClass(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

obj = OpaqueClass(1, 2)
print(obj)
```
```
>>>
<__main__.OpaqueClass object at 0x107880ba8>
```

* 이 결과는 eval 함수에 넘길 수 없으며, 객체의 인스턴스 필드에 대한 정보를 전혀 알려주지 않는다.
* 문제를 해결하는 방법은 다음과 같이 두 가지가 있다.
<br>

* 클래스를 제어할 수 있다면, 직접 `__repr__`이라는 특별한 메서드를 정의해서 객체를 재생성하는 파이썬 표현식을 담은 문자열을 반환하면 된다.
* 다음은 위의 클래스용으로 `__repr__` 함수를 정의한 예다.

```python
class BetterClass(object):
    def __init__(self, x, y):
        # ...

    def __repr__(self):
        return 'BetterClass(%d, %d)' % (self.x, self.y)

obj = BetterClass(1, 2)
print(obj)
```
```
>>>
BetterClass(1, 2)
```

* 클래스 정의를 제어할 수 없을 때는 `__dict__` 속성에 저장된 객체의 인스턴스 딕셔너리를 이용하면 된다.
* 다음은 OpaqueClass 인스턴스의 내용을 출력하는 예다.

```python
obj = OpaqueClass(4, 5)
print(obj.__dict__)
```
```
>>>
{'y': 5, 'x': 4}
```

핵심 정리

* 파이썬 내장 타입에 print를 호출하면 사람이 이해하기는 쉽지만 타입 정보는 숨은 문자열 버전의 값이 나온다.
* 파이썬 내장 타입에 repr를 호출하면 출력할 수 있는 문자열 버전의 값이 나온다. 이 repr 문자열을 eval 내장 함수에 전달하면 원래 값으로 되돌릴 수 있다.
* 포맷 문자열에서 %s는 str처럼 사람이 이해하기 쉬운 문자열을 만들어내며, %r은 repr처럼 출력하기 쉬운 문자열을 만들어낸다.
* `__repr__` 메서드를 정의하면 클래스의 출력 가능한 표현을 사용자화하고 더 자세한 디버깅 정보를 제공할 수 있다.
* 객체의 `__dict__` 속성에 접근하면 객체의 내부를 볼 수 있다.


### 56.

...


### 57. pdb를 이용한 대화식 디버깅을 고려하자

* 누구나 프로그램을 개발할 때 버그를 접한다.
* print 함수를 사용하면 많은 문제의 원인을 추적하는 데 도움이 된다(ref.55. "디버깅 출력용으로는 repr 문자열을 사용하자")
* 문제를 분리하는 또 다른 훌륭한 방법은 문제를 일으키는 특별한 경우를 대비하여 테스트를 작성하는 것이다(ref.56. "unittest로 모든 것을 테스트하자")
* 하지만, 이런 도구만으로는 근본적인 원인을 모두 찾아내지 못한다.
<br>

* 더 강력한 도구가 필요하다면 파이썬에 내장된 대화식 디버거(interactive debugger)를 사용해보자.
* 디버거를 이용하면 프로그램의 상태를 조사하고, 지역 변수를 출력하고, 파이썬 프로그램을 한 번에 한 문장씩 실행할 수 있다.
* 대부분의 다른 프로그래밍 언어에서는 멈추게 할 소스 파일의 줄을 설정한 다음 프로그램을 실행하는 방법으로 디버거를 사용한다.
* 이와 달리 파이썬에서 디버거를 사용하는 가장 쉬운 방법은 프로그램을 수정하여 조사할 만한 문제를 만나기 전에 직접 디버거를 실행하는 것이다.
* 디버거에서 파이썬 프로그램을 실행하는 것과 평소처럼 실행하는 것 사이에는 별다른 차이가 없다.
<br>

* 디버거를 시작하려면 내장 모듈 pdb를 임포트한 후 set_trace 함수를 실행하기만 하면 된다.
   * 보통 이 작업을 한 줄로 처리하여 프로그래머가 # 문자 하나로 주석 처리할 수 있도록 한다.

```python
def complex_func(a, b, c):
    # ...
    import pdb; pdb.set_trace()
```

* 이 문장을 실행하자마자 프로그램은 실행을 멈춘다. 그리고 프로그램을 시작한 터미널은 대화식 파이썬 쉘로 바뀐다.

```
-> import pdb; pdb.set_trace()
(Pdb)
```

* (Pdb) 프롬프트에서 변수의 값을 출력하려면 지역 변수의 이름을 입력하면 된다.
* 내장 함수 locals를 호출하면 모든 지역 변수의 리스트를 볼 수 있다.
* 모듈 임포트, 전역 상태 조사, 새 객체 생성, 내장 함수 help 실행, 심지어 프로그램의 일부를 수정하는 일을 포함해 디버깅을 보조하는 작업은 무엇이든 할 수 있다.
* 게대가 디버거는 실행 중인 프로그램을 더 쉽게 조사할 수 있는 명령 세 개를 제공한다. (아래 명령어는 현재 실행 중인 상태를 파악)
   * `bt`: 현재 실행 호출 스택의 추적 정보를 출력한다. 이 정보는 프로그램의 어느 부분에 있고 pdb.set_trace 트리거 지점에 어떻게 도달했는지 보여준다.
   * `up`: 현재 함수의 호출자 쪽으로 함수 호출 스택의 스코프를 이동한다. 이 동작으로 호출 스택의 상위 레벨에서 지역 변수를 조사할 수 있다.
   * `down`: 함수 호출 스택을 한 단계 낮춘다.
<br>

* 현재 실행 중인 상태를 조사하고 나면 다음과 같은 디버거 명령을 사용하여 좀 더 세밀한 제어로 프로그램 실행을 재개할 수 있다.
   * `step`: 프로그램의 다음 줄까지 실행한 다음 제어를 디버거에게 돌려준다. 다음 줄의 실행이 함수 호출을 포함하고 있다면 디버거는 호출된 함수 안에서 바로 멈춘다.
   * `next`: 현재 함수에서 다음 줄까지 프로그램을 실행한 다음 제어를 디버거에게 돌려준다. 다음 줄의 실행이 함수 호출을 포함하고 있다면 호출된 함수가 반환할 때까지 디버거가 멈추지 않는다.
   * `return`: 현재 함수가 값을 반환할 때까지 프로그램을 실행한 다음 제어를 디버거에 돌려준다.
   * `continue`: 다음 중단점(혹은 set_trace가 다시 호출될 때)까지 프로그램 실행을 계속한다.

#### 핵심 정리
* import pdb; pdb.set_trace() 문으로 프로그램의 관심 지점에서 직접 파이썬 대화식 디버거를 시작할 수 있다.
* 파이썬 디버거 프롬프트는 실행 중인 프로그램의 상태를 조사하거나 수정할 수 있도록 해주는 완전한 파이썬 쉘이다.
* pdb 쉘 명령을 이용하면 프로그램 실행을 세밀하게 제어할 수 있다. 또한 프로그램의 상태를 조사하거나 프로그램 실행을 이어가는 과정을 교대로 반복할 수 있다.


### 58. 

...

### 59. tracemalloc으로 메모리 사용 현황과 누수를 파악하자

* 파이썬, 즉 CPython의 기본 구현은 참조 카운팅(reference counting)으로 메모리를 관리한다.
* 참조 카운팅은 객체의 참조가 모두 해제되면 참조된 객체 역시 정리됨을 보장한다.
* CPython은 자기 참조 객체가 결국 가비지 컬렉션되는 것을 보장하는 사이클 디텍터(cycle detector)도 갖추고 있다.
* 이론적으로는 대부분의 파이썬 프로그래머가 프로그램에서 일어나는 메모리 할당과 해제를 걱정할 필요가 없다는 의미이다.
* 즉, 언어와 CPython 런타임이 자동으로 처리한다.
* 그러나, 실제로 프로그램은 결국 참조 때문에 메모리 부족에 처한다.
* 파이썬 프로그램이 어디서 메모리를 사용하거나 누수를 일으키는지 알아내는 건 힘든 과제이다.
<br>

* 메모리 사용을 디버깅하는 첫 번째 방법은 내장 모듈 gc에 요청하여 가비지 컬렉터에 알려진 모든 객체를 나열하는 것이다.
* gc가 그렇게 정확한 도구는 아니지만 이 방법을 이용하면 프로그램의 메모리가 어디서 사용되는지 금방 알 수 있다.
* 다음과 같이 참조를 유지하여 메모리를 낭비하는 프로그램을 실행해보자. (이 프로그램은 실행 중에 생성된 객체의 수와 할당된 객체들의 샘플을 소량 출력한다.)

```python
# using_gc.py
import gc
found_objects = gc.get_objects()
print('%d objects before' % len(found_objects))

import waste_memory
x = waste_memory.run()
found_objects = gc.get_objects()
print('%d objects after' % len(found_objects))
for obj in found_objects[:3]:
    print(repr(obj)[:100])
```
```
4756 objects before
14873 objects after
<waste_memory.MyObject object at 0x1063f6940>
<waste_memory.MyObject object at 0x1063f6978>
<waste_memory.MyObject object at 0x1063f69b0>
```

* 




























