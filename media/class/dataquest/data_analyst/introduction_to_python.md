
# Introduction to Python

## Python for Data Science: Fundamentals

컴퓨터가 일을 하기 위해서는 instruction이 필요하다. 우리는 a set of instructions를 줘서 컴퓨터에게 일을 시킨다. 이게 programming한다고 표현한다. 다양한 programming language들이 존재한다.

```
or 에서는 True가 우세
True or False or False or False or False => True

and 에서는 False가 우세
True and True and True and False and True => False

and/or가 섞여 있는 경우, 괄호를 넣어 logical error가 발생하지 않도록 한다
(True or False and False)    (x)
((True or False) and False)  (o)
```

파이썬에서 내부적으로 hash() 함수를 사용해서, 딕셔너리의 key는  integer값으로 변환한다. 단, hash() 함수는 리스트나 딕셔너리는 변환하지 못한다. 딕셔너리에서 키가 중복으로 존재할 경우 마지막에 있는 키 정보만 살아남는다.

주의. True 라는 bool 타입의 키가 있고, 1이라는 integer 타입의 키가 있다면, 이들은 중복이다. hash() 함수가 True를 1로 변환시키기 때문이다. 즉, 마지막에 있는 키 정보가 살아남는다.

주의. max()나 min() 함수는 string 타입의 숫자 에게는 적용되지 않는다. 사용하기 전에 항상 모두 integer 타입인지 확인하자.

파이썬에는 read-made 함수 또는 built-in 함수들이 존재한다. 예를 들어, sum(), len(), min(), max(), ...

In programming, errors are known as bugs. The process of fixing an error is commonly known as debugging.

https://docs.python.org/3/ 의 Search page 에서 built-in 함수의 문서를 찾아볼 수 있다.

함수에 여러 개의 리턴 값들이 있으면, 튜플 또는 리스트로 표현하면 된다. 튜플은 그대로, 리스트는 괄호 [ ] 를 감싸준다.



## Python for Data Science: Intermediate

Python has a built-in csv module that can handle the work of opening a CSV for us.

```
from csv import reader    # built-in module
opened_file = open('children.csv')
read_file = reader(opened_file)   # parse (or interpret) the opened_file
children = list(read_file)    # convert the read_file into a list of lists format
children = children[1:]   # remove the first row of the data (=column names)
```

Often when we're cleaning data, we need to replace parts of strings so our data is consistent.

```
strings = ["good!", "morn?ing", "good?!", "morniZZZZng"]
bad_chars = ["!", "?", "Z"]

for row in moma:
    nationality = row[2]
    nationality = nationality.replace("(","")
    nationality = nationality.replace(")","")
    row[2] = nationality
```

replace 함수가 data cleaning의 기초다

```
def strip_characters(string):
    for char in bad_chars:
        string = string.replace(char,"")
    return string
```

The str.title() method returns a copy of the string with the first letter of each word transformed to uppercase (also known as title case).

```
my_string = "The cool thing about this string is that it has a CoMbInAtIoN of UPPERCASE and lowercase letters!"
my_string_title = my_string.title()
print(my_string_title)
>>>
The Cool Thing About This String Is That It Has A Combination Of Uppercase And Lowercase Letters!
```

아래와 같이 전처리를 할 때, empty string에 대해서 int() 함수가 적용되지 않도록 예외처리를 해야 한다. (이런 오류 방지 -> `ValueError: invalid literal for int() with base 10: '')`

```
def clean_and_convert(date):
    # check that we don't have an empty string
    if date != "":
        # move the rest of the function inside
        # the if statement
        date = date.replace("(", "")
        date = date.replace(")", "")
        date = int(date)
    return date

clean_and_convert("")
```

숫자 데이터 중에서 1964.5, 1964 와 같이 자리수가 통일되지 않는 경우가 많다. 통일하고 싶으면 round() 함수를 사용하자.

다음과 같은 방법으로 substring을 찾을 수 있다.

```
if "mike" in "michael":
    print("The substring was found.")
else:
    print("The substring was not found.")
```

다음과 같이 frequency table 을 구축해서 사용하면 좋다.

```
fruit = ['orange', 'orange', 'orange', 'banana',
         'apple', 'banana', 'orange', 'banana',
         'apple', 'banana']

fruit_frequency = {}

for f in fruit:
    if f not in fruit_frequency:
        fruit_frequency[f] = 1
    else:
        fruit_frequency[f] += 1
print(fruit_frequency)

>>>
{
    'orange': 4,
    'banana': 4,
    'apple': 2
}
```

str.format() 함수를 다양한 format specifications룰 추가해서 쓸 수 있다.

```
num = 32.554865

print("I own {}% of the company".format(num))
>>>
I own 32.554865% of the company

print("I own {pct}% of the company".format(pct=num))
>>>
I own 32.554865% of the company

print("I own {pct:.2f}% of the company".format(pct=num))
>>>
I own 32.55% of the company


print("The approximate population of {0} is {1}".format("India",1324000000))
>>>
The approximate population of India is 1324000000

print("The approximate population of India is {0:,}".format(1324000000))
>>>
The approximate population of India is 1,324,000,000
# ',': thousands separator
```


### Object-Oriented Python

지금까지 Procedural Programming 을 배웠다. In its simplest definition, procedural programming involves writing code in a number of sequential steps — and sometimes we combine these steps into commands called functions.

이제 object-oriented programming (OOP) 를 배워보자. Rather than code being designed around sequential steps, it is instead defined around objects. For now, you can think of objects as being closely related to variables
















