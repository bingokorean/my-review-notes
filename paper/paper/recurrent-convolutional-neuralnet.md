# Recurrent Convolutional Neural Network for Text Classification
Review note of this paper is in [here (Presentation File)](https://1drv.ms/p/s!AllPqyV9kKUrj16s-QyaBHBHg70p).

## Summary
* Recurrence와 Convolution 구조를 합친 Recurrent Convolutional Neural Network 제안
* 모델 개발 배경은, (baseline들과) RecursiveNN과 CNN을 개선하기 위함. 특히, CNN을 보완하기 위함인데, CNN과 비교했을 때, convolutional layer 대신 reccurent layer가 들어가는 특징을 가짐 (문맥정보를 Bi-RNN이 잘 포함해준다)
* 모델 내에서는 word representation을 만드는 모듈과 text represention을 만드는 모듈이 내장되어 있음.
   * 문맥을 포함하는 word representation을 만들기 위해 bi-directional rnn사용 (CNN with various window보다는 Bi-RNN이 항상 성능이 더 좋음)
   * max-pooling을 통해 하나의 text vector를 만듦 (예를 들면, word representation이 300D라고 하면, n개의(n-time) word representation들 중에서 각각의 element기준 가장 큰 element 값만 추출한다) -> find the most important semantic factors
* (참고) 논문 내용 중에서 skip-gram을 학습 완료 후 임베딩 e'가 아닌 e를 사용하는 이유로 hierarchical softmax와 같은 speed-up approach 때문이라고 언급하고 있음.
* hyper-parameter tuning은 따로 하지 않고, 기존 연구의 것들을 그대로 사용하거나, 임의로 정해서 사용한다.
* Word representation을 Contextuality로, Text representation을 Compositionality로 연관지어 생각할 수 있음.

## Reference 
Lai, Siwei, et al. "Recurrent Convolutional Neural Networks for Text Classification." AAAI. Vol. 333. 2015.
