# 1장. 파이썬 기반의 머신러닝과 생태계 이해

* [1.1. 머신러닝의 개념](#1.1.-머신러닝의-개념)
* 1.2. 파이썬 머신러닝 생태계를 구성하는 주요 패키지
* [1.3. 넘파이](#1.3.-넘파이)
* [1.4. 데이터 핸들링 - 판다스](#1.4.-데이터-핸들링---판다스)

<br>

## 1.1. 머신러닝의 개념

* 머신러닝(Machine Learning)의 개념은 다양하게 표현할 수 있으나, 일반적으로는 애플리케이션을 수정하지 않고도 데이터를 기반으로 패턴을 학습하고 결과를 예측하는 알고리즘 기법을 통칭합니다.
* 업무적으로 복잡한 조건/규칙들이 다양한 형태로 결합하고 시시각각 변하면서 도저히 소프트웨어 코드로 로직을 구성하여 이들을 관통하는 일정한 패턴을 찾기 어려운 경우에 머신러닝은 훌륭한 솔루션을 제공합니다.
* 머신러닝은 인간의 언어에서 패턴을 규정하기 어려운 문제를 데이터를 기반으로 숨겨진 패턴을 인지해 해결합니다.
* 머신러닝 알고리즘은 데이터를 기반으로 통계적인 신뢰도를 강화하고 예측 오류를 최소화하기 위한 다양한 수학적 기법을 적용해 데이터 내의 패턴을 스스로 인지하고 신뢰도 있는 예측 결과를 도출해 냅니다.
* 데이터 분석 영역은 재빠르게 머신러닝 기반의 예측 분석(Predictive Analysis)으로 재편되고 있습니다.
* 데이터마이닝, 영상 인식, 음성 인식, 자연어처리에서 개발자가 데이터나 업무 로직의 특성을 직접 감안한 프로그램을 만들 경우 난이도와 개발 복잡도가 너무 높아질 수 밖에 없는 분야에서 머신러닝이 급속하게 발전하고 있습니다.
* 머신러닝의 큰 단점은 데이터에 매우 의존적이라는 것입니다. 가비지 인(Garbage In), 가비지 아웃(Garbage out).
* 최적의 머신러닝 알고리즘과 모델 파라미터를 구축하는 능력도 중요하지만 데이터를 이해하고 효율적으로 가공, 처리, 추출해 최적의 데이터를 기반으로 알고리즘을 구동할 수 있도록 준비하는 능력이 더 중요할 수 있습니다.

## 1.3. 넘파이

#### Numpy ndarray 개요

```python
import numpy as np

array1 = np.array([1,2,3])
print('array1 type:',type(array1))
print('array1 array 형태:',array1.shape)

array2 = np.array([[1,2,3],
                  [2,3,4]])
print('array2 type:',type(array2))
print('array2 array 형태:',array2.shape)

array3 = np.array([[1,2,3]])
print('array3 type:',type(array3))
print('array3 array 형태:',array3.shape)
```
```
array1 type: <class 'numpy.ndarray'>
array1 array 형태: (3,)
array2 type: <class 'numpy.ndarray'>
array2 array 형태: (2, 3)
array3 type: <class 'numpy.ndarray'>
array3 array 형태: (1, 3)
```
```python
print('array1: {:0}차원, array2: {:1}차원, array3: {:2}차원'.format(array1.ndim,array2.ndim,array3.ndim))
```
```
array1: 1차원, array2: 2차원, array3:  2차원
```
```python
list1 = [1,2,3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)
```
```
<class 'list'>
<class 'numpy.ndarray'>
[1 2 3] int64
```
```python
list2 = [1, 2, 'test']
array2 = np.array(list2)
print(array2, array2.dtype)
# ['1' '2' 'test'] <U21 
# 숫자형 값 1, 2는 모두 문자열값인 '1', '2'로 변환.
# ndarray는 데이터값이 모두 같은 데이터 타입이어야 함.
# 다른 데이터 타입인 경우, 더 큰 데이터 타입으로 변환됨. (여기선 int형이 유니코드 문자열 값으로 변환)

list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)
# int보다 float이 더 큰 데이터 타입임.
```
```
['1' '2' 'test'] <U21
[1. 2. 3.] float64
```
```python
# 데이터 타입 변경
# 메모리를 절약해야 할 때 보통 이용

array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

array_int1= array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2= array_float1.astype('int32')
print(array_int2, array_int2.dtype)
```
```
[1. 2. 3.] float64
[1 2 3] int32
[1 2 3] int32
```

#### ndarray를 편리하게 생성하기 - arange, zeros, ones































