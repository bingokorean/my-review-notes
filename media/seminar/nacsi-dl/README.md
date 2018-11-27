# NACSI Deep Learning 

2016.4.15~16 (12시간) <br>
서울대학교 제2공학관 <br>
한국인지과학산업협회 (NACSI)

## Contents
* [이론] 딥러닝 AI 기술 [[**노트정리**]](https://1drv.ms/w/s!AllPqyV9kKUrgRNMQiWdmqm-fhlu)
* [실습] [tensorflow-basic](https://github.com/gritmind/review-media/tree/master/seminar/nacsi-dl/tensorflow-basic)
* [실습] [theano-basic](https://github.com/gritmind/review-media/tree/master/seminar/nacsi-dl/theano-basic)

## Summary
* deep neural network는 global search를 포기하고 local optimization을 선택하는 것이고 shallow neural network는 global optimization을 선택하는 것이다. 그런데, deep neural network가 성능이 잘나온다. 실제 문제에서 global search를 하기란 매우 힘들다. 차라리 좀 더 괜찮은 local optima를 선택하는게 더 좋다
* 머신러닝의 trade-off (specialization과 generalization)을 교모하게 잘 이용해야 한다. 예를 들어 fully-connected 구조를 가질 것인지 아니면 spare한 구조를 가질 것인지 결정해야 한다.
* 머신러닝의 근본 원리는 조각을 모두 결합해서 global한 approximation을 만드는 것이다. CNN도 먼저 모두 조각내고 재조립한다. 그리고 Weight sharing을 통해 학습 해야 될 parameter 개수는 줄이면서 다시 팽창하고, dimension reduction한다. 팽창하는 것은 specialization을 유도하는 것이고 줄어드는 것은 generalization을 유도하는 것이다. 즉, Convergence (specialization 유도)와 divergence (generalization 유도)를 반복한다.
* 전통적인 프로그래밍은 분리하고 분류하는 specialization을 잘하는 반면에, 머신러닝은 association, 결합, 융합, 멀티모달 integration하는 것을 잘한다. 문제를 풀기 위해서는 두 가지가 다 필요하다. 왜냐하면 결합만 계속 하면 엉뚱한 그림이 나올 수도 있다. 원본에 충실하면서 거기서 끝나지 않고 새로운 것을 생성하는, 교묘한 결합을 해야 한다. 결국엔 많은 문제들이 이 두 가지를 어떻게 교묘하게 결합하느냐의 문제이다.
