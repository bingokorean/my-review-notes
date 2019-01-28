# Effective PYTHON - 파이썬 코딩의 기술

브렛 슬라킨 지음 <br>
김형철 옮김

### Contents
1.	[파이썬다운 생각](#파이썬다운-생각)



## 파이썬다운 생각

'파이썬다운'이라는 형용사로 파이썬 스타일을 표현. 파이썬 스타일은 컴파일러가 정의하는 것이 아닌 파이썬 개발자들이 수년간 사용하면서 자연스럽게 생겨난 것임. 복잡함보다는 단순함을, 가독성을 극대화하기 위해 명료한 것을 좋아함. (import this를 쳐보자).

### 1. 파이썬 버전을 확인하자

```
import sys
print(sys.version_info)
print(sys.version)
```

파이썬2에서 2to3, six와 같은 도구를 사용하면 파이썬3으로 쉽게 옮길 수 있다.

* 파이썬에는 CPython, Jython, IronPython, PyPy 같은 다양한 런타임 환경이 있다.
* 파이썬 커뮤니티에서 주로 다루는 버전은 파이썬3이므로 새 파이썬 프로젝트는 파이썬3가 좋다.

### 2. PEP 8 스타일 가이드를 따르자

파이썬 개선 제안서(Python Enhancement Proposal #8; PEP 8) <br>
일관성 있는 스타일로 유지보수가 더욱 쉬워지고 가독성도 높아지고 다양한 프로젝트에서 협업도 가능하다.

### 3. bytes, str, unicode 차이점을 알자

** 주의: 파이썬3 중점으로 기술 <br>
파이썬3에서는 bytes (raw 8비트; binary 데이터)와 str (unicode 문자) 두 가지 타입으로 문자 시퀀스를 나타낸다. bytes 인스턴스는 raw 8비트 값을 저장하고, str 인스턴스는 unicode 문자를 저장한다. unicode 문자를 binary 데이터로 표현하는 방법은 많다. (ex. UTF-8 인코딩). 허나, str 인스턴스는 연관된 binary 인코딩이 없다 (아예 변환하는 것). 따라서, unicode 문자를 binary 데이터로 변환하려면 encode 함수를, binary 데이터를 unicode 문자로 변환하려면 decode 함수를 사용해야 한다.

파이썬 프로그래밍할 때 외부에 제공할 인터페이스에서는 unicode를 encode하고 decode해야 한다 (즉, 바이너리 인코딩). 프로그램 핵심 부분에서는 unicode 문자 타입(ex. str)을 사용하고, 문자 인코딩에 대해서는 어떤 가정도 하지 말아야 한다. 그러기 위해서 다음 두 가지 헬퍼 함수가 필요하다.

```
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

파이썬3에서 bytes와 str 인스턴스는 심지어 빈 문자열도 같지 않으므로 함수에 넘기는 문자열의 타입을 더 신중하게 처리해야 한다. <br>
파이썬3에서 내장 함수 open이 반환하는 파일 핸들을사용하는 연산은 기본적으로 UTF-8 인코딩을 사용한다. (파이썬2에서 파일 연산은 바이너리 인코딩이다)
따라서, 파이썬3에서 .bin 파일을 open할 때 'w'가 아닌 'wb'(바이너리-쓰기) 모드를, 'r'이 아닌 'rb'를 사용해야 한다.

* 파이썬3에서는 bytes는 8비트 값을 저장하고, str은 유니코드 문자를 저장한다. (>나+와 같은 연산자에 bytes와 str 인스턴스를 함께 사용할 수 없다)
* 헬퍼 함수를 사용해서 처리할 입력값이 원하는 문자 시퀀스 타입(8비트 값, UTF-8 인코딩 문자, 유니코드 문자 등)으로 되어 있게 한다.
* 바이너리 데이터를 파일에서 읽거나 쓸 때는 파일을 바이너리 모드('rb' 혹은 'wb')로 오픈한다.

### 4. 복합한 표현식 대신 헬퍼 함수를 사용하자

파이썬은 간결한 문법을 이용하면 많은 로직을 표현식 한 줄로 쉽게 작성할 수 있다. 예를 들어 URL에서 쿼리 문자열을 디코드해야 한다고 하자. 다음 예에서 각 쿼리 문자열 파라미터는 정수 값을 표현한다.

```
from urllib.parse import parse_qs
my_values = parse_qs('red=5&blue=0&green=', keep_blank_values=True)
```

쿼리 문자열 파라미터에 따라 값이 여러 개 있을 수도 있고 값이 한 개만 있을 수도 있으며, 파라미터는 존재하지만 값이 비어 있을 수 있고, 파라미터가 아예 빠진 경우도 있다. 파라미터가 없거나 비어 있으면 기본값으로 0을 할당하면 좋다. 다음 처리 방식을 보자.

```
red = my_values.get('red', [''])[0] or 0  # '5'
green = my_values.get('green', [''])[0] or 0 # 0
opacity = my_values.get('opacity', [''])[0] or 0 # 0
```

숫자 변환을 위해 `red = int(my_values.get('red', [''])[0] or 0)`으로 할 수도 있다. 이들의 코드를 읽기는 쉽지 않다. if/else 문이 훨씬 더 직감적일 것이다.

```
green = my_values.get('green', [''])
if green[0]:
    green = int(green[0])
else:
    green = 0
```

하지만 이를 반복적으로 사용하기 보다는 다음과 같은 헬퍼 함수로 처리하면 어떨까?

```
def get_first_int(values, key, default=0):
    found = values.get(key, [''])
    if found[0]:
        found = int(found[0])
    else:
        found = default
    return found
```

`green = get_first_int(my_values, 'green')` 이렇게 헬퍼 함수를 이용하면, or를 사용한 복잡한 표현식이나 if/else 조건식 버전보다 호출 코드가 훨씬 더 간결하고 명확해진다.

표현식이 복잡해지기 시작하면 최대한 빨리 해당 표현식을 작은 조각으로 분할하고 로직을 헬퍼 함수로 옮기는 방안을 고려해야 한다. 무조건 짧은 코드를 만들기보다는 가독성을 선택하는 편이 낫다. 이해하기 어려운 복잡한 표현식에는 파이썬의 함축적인 문법을 사용하면 안 된다.

* 파이썬의 문법은 한 줄짜리 표현식을 쉽게 작성할 수 있지만 코드가 복잡해지고 읽기 어려워진다.
* 복잡한 표현식은 헬퍼 함수로 옮기는 게 좋다. 특히, 같은 로직을 반복해서 사용한다면 헬퍼 함수를 사용하자.
* if/else 표현식을 이용하면 or나 and 같은 불 연산자를 사용할 때보다 읽기 수월한 코드를 작성할 수 있다.

### 5. 시퀀스를 슬라이스하는 방법을 알자

파이썬은 시퀀스를 slice해서 부분집합에 접근할 수 있도록 해준다. 가장 간단한 슬라이싱 대상은 내장 타입인 list, str, btyes이다. `__getitem__`과 `__setitem__`이라는 특별한 메서드를 구현하는 클래스에도 slicing을 적용할 수 있다. slicing 기본 문법 형태는 somelist[start:end]이며, 여기서 start 인덱스는 포함되고 end 인덱스는 제외된다.

list의 처음부터 slice할 때는 보기 편하게 인덱트 0을 생략하고, list의 끝까지 slice할 때도 마지막 인덱스는 넣지 않아도 되므로 생략한다.
```
assert a[:5] == a[0:5]
assert a[5:] == a[5:len(a)]
```
리스트의 끝을 기준으로 오프셋을 계산할 때는 음수로 slice하는 게 편하다. 
```
a = [1,2,3,4,5,6,7,8,9]
a[-4:] # 뒤에서 4번째까지 추출해라.
>>> [6,7,8,9] 
a[-4:-1] # 이런 형태는 조심
>>> [6,7,8]
```
slicing은 start와 end 인덱스가 리스트의 경계를 벗어나도 적절하게 처리한다. 덕분에 입력 시퀀스에 대응해 처리할 최대 길이를 코드로 쉽게 설정할 수 있다. 이와 대조적으로 같은 인덱스를 직접 접근하면 예외가 발생한다.
```
first_twenty_items = a[:20]
>>> [1,2,3,4,5,6,7,8,9]
a[20]
>>> IndexError: list index out of range
```

[NOTE] 리스트의 인덱스를 음수로 지정하면 slicing이 뜻밖의 결과를 얻기도 한다. 예를 들어, somelist[-n:]이라는 구문은 somelist[-3:]처럼 n이 1보다 클 때는 제대로 동작하지만, n이 0이어서 somelist[-0:]이 되면 원본 리스트의 복사본을 만든다.

slicing 결과는 완전히 새로운 리스트이지만, 원본 리스트에 들어 있는 객체에 대한 참조는 유지된다. 하지만, slice한 결과를 수정해도 원본 리스트에 아무런 영향을 미치지 않는다. 

할당에 사용하면 slice는 원본 리스트에서 지정한 범위를 대체한다. `a, b = c[:2]` 같은 튜플 할당과 달리 slice 할당은 길이가 달라도 된다. 할당받은 slice 앞뒤 값은 유지된다. 리스트는 새로 들어온 값에 맞춰 늘어나거나 줄어든다.
```
a[2:7] = [99,22,14]
print(a)
>>> [1,2,99,22,14,8,9]
```
시작과 끝 인덱스를 모두 생략하고 slice하면 원본 리스트의 복사본을 얻는다
```
b = a[:]
assert b == a and b is not a
```
slice에 시작과 끝 인덱스를 지정하지 않고 할당하면 (새 리스트를 할당하지 않고) slice의 전체 내용을 참조 대상의 복사본으로 대체한다. (즉, 주소 복사 실시)
```
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

```
a = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
odds = a[::2]
evens = a[1::2]
print(oods)
print(evens)
>>> ['red', 'yellow', 'blue']
>>> ['orange', 'green', 'purple']
```

문제는 stride 문법이 종종 예상치 못한 동작을 해서 버그를 만들어내기도 한다. 예를 들어, 파이썬에서 바이트 문자열을 역순으로 만든느 일반적인 방법은 stride -1로 문자열을 slice하는 것이다. 문제는 바이트 문자열이나 아스키 문자에는 잘 동작하지만, UTF-8 바이트 문자열로 인코드된 유니코드 문자에는 원하는 대로 동작하지 않는다. 
```
w = '漢字'
x = w.encode('utf-8')
y = x[::-1]
z = y.decode('utf-8')
>>> 
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9d in 
position 0: invalid start byte
```

-1을 제외한 음수 값으로 stride를 지정하면 어떨까? 2::2는 무슨 뜻일까?

요점은 slicing 문법의 stride 부분이 매우 혼란스러울 수 있다는 점이다. 대괄호 안에 숫자가 세 개나 있으면 빽빽해서 읽기 어렵고 start와 end 인덱스가 stride와 연계되어 어떤  작용을 하는지 분명하지 않다. 특히 stride가 음수인 경우는 더욱 그러하다.

이러한 문제를 방지하기 위해 stride를 start, end 인덱스와 함께 사용하지 말아야 한다. stride를 사용해야 한다면 양수 값을 사용하고 start와 end 인덱스는 생략하는 게 좋다. stride를 꼭 start와 end 인덱스와 함께 사용해야 한다면 stride를 적용한 결과를 변수에 할당하고, 이 변수를 slice한 결과를 다른 변수에 할당해서 사용하자.
```
a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
b = a[::2] # ['a', 'c', 'e', 'g']
c = b[1:-1]
```
slicing부터 하고 striding을 하면 데이터의 shallow copy가 추가로 생긴다. 첫 번째 연산은 결과로 나오는 slice 크기를 최대한 줄여야 한다. 프로그램에서 두 과정이 필요한 시간과 메모리가 충분하지 않다면 내장 모듈 itertools의 islice 메서드를 사용하자. islice 메서드는 start, endm, stride에 음수 값을 허용하지 않는다.

* 한 slice에 start, end, stride를 지정하면 혼란스러울 수 있다.
* slice에 start와 end 인덱스 없이 양수 stride 값을 사용하자. 음수 stride 값은 가능하면 피하는 게 좋다.
* 한 slice에 start, end, stride를 함께 사용하는 상황은 피하자. 파라미터 세 개를 사용해야 한다면 두 개(slice, 다른 하나는 stride)를 사용하거나 내장 모듈 itertools의 islice를 사용하자.



