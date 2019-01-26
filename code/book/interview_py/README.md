# Elements of Programming Interviews in Python

## Contents

_The Interview_
* ~~Ready, Strategy, Conduct an Interview~~

_Data Structures and Algorithms_ 
* Primitive Types
* [Arrays](https://github.com/gritmind/review/blob/master/code/book/interview_py/notes/arrays.md)
* Strings
* Linked Lists
* Stacks and Queus
* Binary Trees
* Heaps
* Searching
* Hash Tables
* Sorting
* Binary Search Trees
* Recursion
* Dynamic Programming
* Greedy Algorithms and Invariants
* Graphs
* Parallel Computing

_Domain Specific Problems_ 
* ~~Design Problems~~
* ~~Language Questions~~
* ~~Object-Oriented Design~~
* ~~Common Tools~~

_The Honors Class_
* ~~Honors Class~~



~~~~~~{.python}
def hi(): ldkjflaksdjflkdsjflkdsajflksdjlkfjsldkflksjdflksalklskjflksdfjlkdsajdsfds sf
    print('hi')
~~~~~~


```python
def hi(): ldkjflaksdjflkdsjflkdsajflksdjlkfjsldkflksjdflksalklskjflksdfjlkdsajdsfds sf
    print('hi')
```

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
