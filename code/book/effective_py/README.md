# Effective PYTHON - 파이썬 코딩의 기술

브렛 슬라킨 지음 <br>
김형철 옮김

### Contents
1.	[파이썬다운 생각](#파이썬다운-생각)



## 파이썬다운 생각

'파이썬다운'이라는 형용사로 파이썬 스타일을 표현. 파이썬 스타일은 컴파일러가 정의하는 것이 아닌 파이썬 개발자들이 수년간 사용하면서 자연스럽게 생겨난 것임. 복잡함보다는 단순함을, 가독성을 극대화하기 위해 명료한 것을 좋아함. (import this를 쳐보자).

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

파이썬 개선 제안서(Python Enhancement Proposal #8; PEP 8) <br>
일관성 있는 스타일로 유지보수가 더욱 쉬워지고 가독성도 높아지고 다양한 프로젝트에서 협업도 가능하다.

### 3. bytes, str, unicode 차이점을 알자

** 주의: 파이썬3 중점으로 기술 <br>
파이썬3에서는 bytes (raw 8비트; binary 데이터)와 str (unicode 문자) 두 가지 타입으로 문자 시퀀스를 나타낸다. bytes 인스턴스는 raw 8비트 값을 저장하고, str 인스턴스는 unicode 문자를 저장한다. unicode 문자를 binary 데이터로 표현하는 방법은 많다. (ex. UTF-8 인코딩). 허나, str 인스턴스는 연관된 binary 인코딩이 없다 (아예 변환하는 것). 따라서, unicode 문자를 binary 데이터로 변환하려면 encode 함수를, binary 데이터를 unicode 문자로 변환하려면 decode 함수를 사용해야 한다.

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

파이썬3에서 bytes와 str 인스턴스는 심지어 빈 문자열도 같지 않으므로 함수에 넘기는 문자열의 타입을 더 신중하게 처리해야 한다. <br>
파이썬3에서 내장 함수 open이 반환하는 파일 핸들을사용하는 연산은 기본적으로 UTF-8 인코딩을 사용한다. (파이썬2에서 파일 연산은 바이너리 인코딩이다)
따라서, 파이썬3에서 .bin 파일을 open할 때 'w'가 아닌 'wb'(바이너리-쓰기) 모드를, 'r'이 아닌 'rb'를 사용해야 한다.

* 파이썬3에서는 bytes는 8비트 값을 저장하고, str은 유니코드 문자를 저장한다. (>나+와 같은 연산자에 bytes와 str 인스턴스를 함께 사용할 수 없다)
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

숫자 변환을 위해 `red = int(my_values.get('red', [''])[0] or 0)`으로 할 수도 있다. 위를 포함해 이들의 코드를 읽기는 쉽지 않다. if/else 문이 훨씬 더 직감적일 것이다. 하지만 이를 반복적으로 사용하기 보다는 다음과 같은 헬퍼 함수로 처리하면 어떨까?

```python
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




