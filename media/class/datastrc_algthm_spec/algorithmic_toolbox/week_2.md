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

### PRoblem Overview

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

뻗어나가는 tree를 언뜻보면, 같은 node들이 보이는데, 이 node들을 다시 재계산할 필요가 없지 않나 생각이 든다. 이 점을 이용하면 more efficient한 알고리즘을 만들 수 있을 것 같다.

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
create an array F[0...n]      # 이 array를 위엥서 written down할 공책이라고 생각하면 된다.
F[0] = 0
F[1] = 1
for i from 2 to n:            # 그냥 1 ~ n 까지의 피보나치 숫자를 공책에 written down하는 것과 같다.
    F[i] = F[i-1] + F[i-2]
return F[n]                   # 해당(마지막) n 번째의 피보나치 숫자를 반환하면 된다.
```

이 알고리즘의 T(n)은 3 (at the beginning) + 1 (last return line) + 2(n-1)(for-loop) 으로 총 합 2n + 2 이다. 이제 T(100)은 202로 굉장히 적은 line 개수가 측정된다. 이 알고리즘의 핵심은 '저장 또는 기억'을 통한 똑같은 계산을 방지하는 것이 아닐까 한다.

피보나치 숫자가 주는 스토리는 the right algorithm makes all the difference 한다는 점이다. 똑같은 문제라 할 지라도 죽기 전까지 끝나지 않는 알고리즘과 눈깜빡하면 끝나는 알고리즘이 있다는 것을 알고 있어야 한다.

## Greatest Common Divisor















