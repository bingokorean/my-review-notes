# Week 2. Algorithmic Warm-up

Learning Objectives
* Estimate the running time of an algorithm
* Practice implementing efficient solutions
* Practice solving programming challenges
* Implement programs that are several orders of magnitude faster than straightforward programs

## Why Study Algorithms?

파일 옮기기, 특정 텍스트 검색하기 등 어떤 알고리즘을 사용할 지 쉽게 알 수 있는 문제가 있지만, optimal path 찾기, 문서 간 similarity 측정하기 등과 같이 조금 더 복잡한 문제들은 보통 어떤 알고리즘을 사용할 지 쉽게 감이 오지 않는다. 복잡한 문제의 솔루션을 찾았다 하더라도 심플한 알고리즘의 경우 too slow한 문제를 가지기도 한다. 심지어 효율적인 알고리즘을 구현했다 하더라도 further optimization할 여지가 많은 게 사실이다. 

대표적인 복잡한 문제는 aritifial intelligence problem을 지칭한다. 이러한 문제는 자연어를 이해하기, 사진 인식하기, 체스 게임하기 등과 같이  hard to even clearly state 한 특징이 있다 (문제를 정의하기 뭔가 애매하다는 뜻이다). 이 코스에서는 이러한 복잡한 문제 자체에 집중하는 것이 아닌 속도와 효율성을 연구하는 알고리즘에 집중할 것이다 (인공지능 문제는 기본 알고리즘 지식이 기본으로 갖춰져 있어야 한다).

Focus on algorithms problems
* Clearly formulated (like clear mathematical problem)
* Hard to do efficiently

몇몇 문제는 immediately 수학 문제는 아니지만, 수학 문제로 clearly what it is 로 해석할 수 있다. 그러나 쉽지는 않을 것이다.

Cover two algorithm problems
* Fibonacci Numbers
* Greatest Common Divisors

왜 위의 문제를 선택했나? 두 문제는 pretty straightforward algorithm 이다. 즉, Problem Statement -> Algorithm 간의 프로세스가 명확하다. 문제를 정의하고 알고리즘으로 해석하기 명확하다. 물론, very straightforward algorithm 은 far, far slow 하다. 평범한 입력을 넣어도 몇 천년이 걸릴 수 있다. 물론, 현실적으로 그대로 사용할 순 없고, better solution을 찾을 것이다.

## Fibonacci Numbers

Fibonacci numbers는 그냥 recursive 특징을 가지는 sequence of natural numbers이다. 이탈리아 수학자가 rabbit population을 위한 수학적 모델을 Fibonacci Numbers로 만들었다 - How many pairs of rabbits you have after n generations. 

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_1.PNG" width="40%" height="40%"></p>

피보나치 시퀀스는 0,1,1,2,3,5,8,13,21,34,... 로 구성되고, 다음과 같이 명백하게 수학적으로 표현할 수 있다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_2.PNG" width="40%" height="40%"></p>

n 이 증가할수록 기하급수적으로 값이 커진다. <br>
F_20 = 6765, F_50 = 12586269025, F_100 = 354224848179261915075, F_500 = 139423224561697880139724382870407283950070256587697307264108962948325571622863290691557658876222521294125

피보나치 숫자는 매우 rapid하게 grow하므로 알고리즘으로 해결해야 한다. 우리는 다음과 같은 입력과 출력을 가지는 컴퓨팅 문제를 해결하면 된다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_3.PNG" width="40%" height="40%"></p>

### Naive Algorithm

가장 naive한 알고리즘은 recursive을 활용한 알고리즘이다.

```
FibRecurs(n)
if n <= 1:
   return n
else:
    return FibRecurs(n-1) + FibRecurs(n-2)
```

위의 simple 알고리즘은 정확하게 잘 동작할 것이다. 하지만, 우리는 efficient한 지도 체크해야 한다. 이 알고리즘이 얼마나 오래 걸릴까? Running time을 **T(n)** - number of lines of code executed by FibRecurs(n)  이라고 정의하고, 대략적으로 T(n)을 통해 얼마나 오래 걸릴지 가늠해볼 수 있다. 예를 들어, T(n)은 3 + T(n-1) + T(n-2) 이다. 다음과 같이 formula하게 정의할 수 있다. 오리지널 피보나치 formula와 비슷하다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_4.PNG" width="40%" height="40%"></p>

보통 출력값인 F_n 보다 실제 계산해야되는 라인 수인 T(n) 이 크다. n=100 일 때, 1GHz의 컴퓨팅으로 T(100)(=엄청난 양의 코드 라인 수)을 계산하면 대략적으로 56,000 years가 소요된다.

왜 이 알고리즘이 slow한가? recursive call하는 big tree이기 때문이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_5.PNG" width="70%" height="70%"></p>

뻗어나가는 tree를 언뜻보면, 같은 node들이 보이는데, 이 node들을 다시 재계산할 필요가 있지 않나 생각이 든다. 이 점을 이용하면 more efficient한 알고리즘을 만들 수 있을 것 같다.

### Efficient Algorithm

efficient 알고리즘을 구현하기 위한 아이디어를 얻는 방법으로 by hand로 일일이 계산해보는 방법이 있다. 

```
0, 1, 1, 2, 3, 5, 8

0 + 1 = 1
1 + 1 = 2 
1 + 2 = 3
2 + 3 = 5
3 + 5 = 8
```

이전에 계산한 모든 결과값들을 written down했기 때문에, 여기서는 recursive하게 계산할 필요가 없어졌다. 이 점을 알고리즘에 적용해보자.

```
FibList(n)
create an array F[0...n]      # 이 array를 위에서 written down할 공책이라고 생각하면 된다.
F[0] = 0
F[1] = 1
for i from 2 to n:            # 그냥 1 ~ n 까지의 피보나치 숫자를 공책에 written down하는 것과 같다.
    F[i] = F[i-1] + F[i-2]
return F[n]                   # 해당(마지막) n 번째의 피보나치 숫자를 반환하면 된다.
```

이 알고리즘의 T(n)은 3 (at the beginning) + 1 (last return line) + 2(n-1)(for-loop) 으로 총 합 2n + 2 이다. 이제 T(100)은 202로 굉장히 적은 line 개수가 측정된다. 이 알고리즘의 핵심은 '저장 또는 기억'을 통한 똑같은 계산을 방지하는 것이 아닐까 한다.

피보나치 숫자가 주는 스토리는 the right algorithm makes all the difference 한다는 점이다. 똑같은 문제라 할 지라도 죽기 전까지 끝나지 않는 알고리즘과 눈깜빡하면 끝나는 알고리즘이 있다는 것을 알고 있어야 한다.

## Greatest Common Divisor

GCD(Greatest Common Divisor) 문제란 어떤 분수 a/b 를 simplest form으로 변환하는 문제이다. 일반적인 방법은 분자(numerator)와 분모(denominator)를 d로 일괄적으로 나누면 된다. 여기서 GCD 문제는 추가적인 조건을 충족해야 한다. 
   * d를 a와 b에 일괄적으로 나눈다 (딱 나눠 떨어져야 한다; 출력값이 integer)
   * 최대한 가장 큰 d를 찾는다

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_6.PNG" width="40%" height="40%"></p>

GCD는 Number Theory에서 매우 중요한 컨셉을 가진다 - study of prime numbers, factorization, ...

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_7.PNG" width="50%" height="50%"></p>

GCD가 Number Theory에서 매우 중요하기 때문에 GCD를 계산하는 것이 cryptography에서 중요한 문제이다 - secure online banking, ... 이처럼 중요하기 때문에 GCD를 알고리즘으로 풀려고 한다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_8.PNG" width="40%" height="40%"></p>

gcd(10,4)와 같이 small number에 대해서는 쉽게 알고 있다. 하지만, 우리는 gcd(3918848, 1653264)와 같이 large number에 대해서 다루고자 한다.

### Naive Algorithm

그냥 순차적으로 처음부터 모두 계산해보는 naive한 방법이 있다.

```
Function NaiveGCD(a,b)
best = 0
for d from 1 to a+b:
   if d|a and d|b:
       best = d
return best
```

물론 느리다. 대략적인 runtime은 a + b이다. 특히, 20 digit number 이상되면 매우, 매우 느려진다.

### Efficient Algorithm

better 알고리즘을 찾기 위해서 something interesting about the structure of the solution. 이 점이 문제를 simplify해줄 수 있다. 여기서는 key lemma를 아는 것이 중요하다. 명제를 충족하는 부명제라고 보면 될 것 같다. 문제 관점을 좀 더 쉽게 보거나 다양하게 볼 수 있고 이를 알고리즘에 반영할 수 있다. 

Key Lemma는 다음과 같다. 나머지(remainder)를 활용한 것이고 증명도 쉽게 할 수 있다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/class/datastrc_algthm_spec/algorithmic_toolbox/images/week_2_9.PNG" width="40%" height="40%"></p>

이 특징을 알고리즘에 적용해보자.

```
Function EuclidGCD(a, b)
if b = 0:                              # 탈출 조건이다. 
   return a
a' = the remainder when a is diviced by b
return EuclidGCD(b, a')                 # a를 a로 교체하는 것뿐만 아니라 b와 자리를 교체하고, recursive하게 함수 콜을 한다.
```

예시를 살펴보자.

```
gcd(3918848, 1653264)
= gcd(1653264, 612320)
= gcd(612320, 428624)
= gcd(428624, 183696)
= gcd(183696, 61232)
= gcd(61232, 0)
= 61232.
```

right answer를 찾기까지 6 step 밖에 걸리지 않았다. 만약에, native 알고리즘을 사용했더라면 5 million step 정도 소요될 것이다. 

이 알고리즘이 잘 동작하는 이유를 runtime 관점에서 살펴보자. 
   * Each step reduces the size of numbers by about a factor of 2
   * Takes about log(ab) steps
   * GCDs of 100 digit numbers takes about 600 steps
   * Each step a single division

## Big-O Notation

### Computing Runtimes

Computing runtime과 프로그램이 얼마나 오래걸리는 지에 대한 이해를 가지도록 한다. 지금까지 line of codes 수로 대략적으로 컴퓨팅 시간을 가늠했다. lines of code로 computing runtime을 측정한다는 것의 전제는 모든 명령어가 동일한 명령어라는 것이다. 그리고 우리가 보는 코드에서 한 줄이 어셈블리어로 분해해서 보면 여러 줄이 된다. 과연, code line 개수로 computing runtime을 계산할 수 있을까? 정확한 방법이 필요하다.

근본적으로 우리가 원하는 바는 실제 컴퓨터가 프로그램을 돌리는 데 걸리는 시간을 측정하면 된다. 사실, 실제로 걸리는 시간을 측정하는 것은 매우 거대한 작업이다 (컴퓨터 스피드, 시스템 아키텍쳐, 메모리 구조 등을 완벽히 이해해야 한다). 알고리즘을 평가하는데 정말 디테일하고 정확하게 실제로 걸리는 시간을 우리가 알 필요가 있을까?

우리의 목표는...
* Measure runtime without knowing these details
* Get results that work for large inputs

### 



