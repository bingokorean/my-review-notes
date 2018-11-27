# 파이썬과 오픈소스를 활용한 딥러닝


2016.09.05~09 (34시간) <br>
멀티캠퍼스, 고태훈 <br>

All notes for this tutorial is in [**HERE**](https://1drv.ms/w/s!AllPqyV9kKUrgnJel5cPmhSkMPkO)

기계학습 이론 및 PYTHON 실습

## Contents
1. Introduction
2. Python Setting
3. Linear Regression
4. Logistic Regression
5. K-neareast neighbors (KNN)
6. Naive Bayes
7. Decision tree: Classification and regression tree (CART)
8. Bagging and Random forest
9. Support Vector Machine
10. Introduction to Deep Learning

## Summary

__Naive Bayes Classifier__
 - 주어진 데이터를 어떤 클래스로 분류하는 분류문제를 확률모델로 풀어보자. 조건부확률 p(C|**X**)로 문제 정의를 한다. 
하지만, 하나의 데이터는 n dimensional (continuous/discrete) features로 표현되기 때문에 확률로 나타내기(inference) 쉽지 않다. (만약 n dimension이 Class의 dimension보다 작고, 각각 binary discrete feature라고 한다면 p(C|**X**)를 곧바로 구하는 것이 쉽다 그런데 그런 경우는 실제 문제에서 거의 없다)
 - 우리의 문제를 확률로 나타내기(inference) 위해서 Bayes Rule을 사용해 조건부 확률, p(C|**X**)의 조건을 뒤집어 p(**X**|C)로 문제를 풀 수 있도록 한다. (여기서 조건부를 뒤집는다 하더라도 문제 정의가 바뀌지 않는다. 샘플 스페이스 (확률의 분모값)만 일정하다고 한다면 데이터와 클래스 사이의 연결 결합도만 측정하면 된다.)
  - 추가적으로 데이터의 feature들은 서로 독립적이다라고 가정을 해서 시퀀스 확률이 아닌 atomic확률들의 곱으로 나타내어 확률을 inference하기 또 더 쉽게 만든다. 결국 이러한 작업들이 모두 제한된 데이터셋에서 확률로 inference하기 위한 것이다. 
  - Feature가 discrete한 경우는 count-based로 확률을 inference할 수 있고, feature가 continuous한 경우에는 probability density estimation (e.g. normal distribution)을 사용해서 확률을 inference할 수 있다.


__Bias and Variance__
 - 기계학습 모델이 샘플데이터에 잘 fitted되었는지 확인하는 척도로 Bias와 variance를 사용한다. 
 - High bias이거나 high variance이면 underfitting / overfitting을 유발하여 성능 저하를 야기한다.
 - 기본적으로 모델 capcity 문제이지만 variance의 경우는 데이터 관점에서도 해석할 수 있다.
 - high bias는 모델 capcity가 낮아서 발생한다.
 - high variance는 모델 capacity가 높아서 발생하고 샘플된 데이터에 내재되어 있는 변동(fluctuation)에 의해 발생한다.
 - high bias 해결방법은 변수를 늘리던가 더 복잡한 모델을 선택한다. 단, 데이터를 늘리거나 줄이는 방법은 도움이 크게 되지 않는다.
 - high variance 해결방법은 (질좋은) 데이터를 늘린다. 또는 변수를 더 적게 두던지 좀 더 간단한 모델을 선택한다
 - 중요한 것은 이 둘이 trade-off 관계에 있기 때문에 서로의 방향에서 극복하는 것도 중요하지만 최적의 합의점을 찾는 것이 가장 중요하다
