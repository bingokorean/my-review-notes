# reddit


## When should we avoid classes? 

[[Link](https://www.reddit.com/r/Python/comments/7nn912/when_should_we_avoid_classes/)] 많은 파이썬 프로그래머들이 small project임에도 class를 많이 사용한다. Class를 항상 사용하는 것이 좋은 것인지? 언제 사용하는 것이 좋은지..?

* Rule of Thumb - "if I need to preserve some state between function calls I will use class with this function as a method"
   * 함수콜을 하는 중에 무언가를 유지하고 있어야 한다면... class를 사용한다.
* Classes are for combining related data and functions that act on that data.
   * 프로그램에는 다양한 data와 function이 있는데, class는 right data와 함께 right function을 사용할 수 있도록 해준다. 이러한 기능은 코드가 증가할 때 코드를 structured 하게 만들어 준다.
   * collections + functions -> classes.
* A class can encapsulate both functionality and data. (Then we can make instances of this class, each holding their own data and interacting with the outside (or internally) via methods.) 
   * 만약, 오직 functionality를 encapsulate하고 싶으면 그냥 fucntion을 사용한다.
   * 파이썬에 없는 data type을 표현하고 싶을 때 class를 사용한다.
