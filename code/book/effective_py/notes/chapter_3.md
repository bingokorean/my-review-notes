## 3. 클래스와 상속

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
