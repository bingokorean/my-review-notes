# Primitive Types

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






