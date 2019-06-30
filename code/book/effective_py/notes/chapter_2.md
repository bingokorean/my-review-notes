## 2. 함수

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

파이썬은 for x in foo 같은 문장을 만나면 실제로 iter(foo)를 호출한다. 그러면 내장 함수 iter는 특별한 메서드인 foo.__iter__를 호출한다. __iter__ 메서드는 (__next__라는 특별한 메서드를 구현하는) iterator 객체를 반환해야 한다. 마지막으로 for 루프는 iterator를 모두 소진할 때까지 (그래서 StopIteration 예외가 발생할 때까지) iterator 객체에 내장 함수 next를 계속 호출한다.

복잡해 보이지만 사실 클래스의 __iter__ 메서드를 generator로 구현하면 이렇게 동작하게 만들 수 있다. 다음은 데이터를 담은 파일을 읽는 iterable 컨테이너 클래스이다. 새로 정의한 컨테이너 타입은 원래의 함수에 수정을 가하지 않고 넘겨도 제대로 동작한다.

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

이 코드가 동작하는 이유는 normalize의 sum 메서드가 새 iterator 객체를 할당하려고 ReadVisits.__iter__를 호출하기 때문이다. 숫자를 정규화하는 for 루프도 두 번째 iterator 객체를 할당할 때 __iter__를 호출한다. 두 iterator는 독립적으로 동작하므로 각각의 순회 과정에 모든 입력 데이터 값을 얻을 수 있다. 이 방법의 유일한 단점은 입력 데이터를 여러 번 읽는다는 점이다.

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

선택적인 위치 인수(이런 파라미터 이름을 관례적으로 *args라고 해서 종종 'star args'라고도 한다)를 받게 만들면 함수 호출을 더 명확하게 하고, 보기에 방해가 되는 요소를 없앨 수 있다. 예를 들어 디버그 정보 몇 개를 로그로 남긴다고 하자. 만약, 인수의 개수가 고정되어 있다면 메시지와 값 리스트를 받는 함수가 필요할 것이다. (리스트 자료 구조로 가변의 인자를 받을 수 있도록 해준다.)

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

*args를 받는 함수는 인수 리스트에 있는 입력의 수가 적당히 적다는 사실을 아는 상황에서는 좋은 방법이다. 이런 함수는 많은 리터럴이나 변수 이름을 한꺼번에 넘기는 함수 호출에 이상적이다. 주로 개발자들을 편하게 하고 코드의 가독성을 높이는 데 사용한다. (즉, *args를 쓴다면 입력의 수가 적당히 적다는 뜻이 내포되어 있다)

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

위 코드의 문제는 두 번째 호출이 sequence 인수를 받지 못해서 7을 message 파라미터로 사용한다는 점이다. (함수 정의는 새로운 버전인데 깜빡하고 예전 버전의 함수 호출을 사용할 수도 있으니...) 이런 버그는 코드에서 예외를 일으키지 않고 계속 실행되므로 발견하기가 극히 어렵다. 이런 문제가 생길 가능성을 완전히 없애려면 *args를 받는 함수를 확장할 때 키워드 전용 인수를 사용해야 한다 (ref.21).

* def 문에서 *args를 사용하면 함수에서 가변 개수의 위치 인수를 받을 수 있다
* *연산자를 사용하면 시퀀스에 들어 있는 아이템을 함수의 위치 인수로 사용할 수 있다
* Generator와 *연산자를 함께 사용하면 프로그램이 메모리 부족으로 망가질 수도 있다
* *args를 받는 함수에 새 위치 파라미터를 추가하면 정말 찾기 어려운 버그가 생길 수도 있다.

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

[Note] 이런 선택적인 키워드 인수를 사용하면 *args를 인수로 받는 함수에서 하위 호환성을 지키기 어렵다(ref.18). 더 좋은 방법은 키워드 전용 인수(ref.21)를 사용하는 것이다.

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
