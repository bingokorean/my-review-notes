# Convolutional Neural Networks for Sentence Classification

[Here (presentation file)](https://1drv.ms/p/s!AllPqyV9kKUrj2BQLRfIToSKPFHB) is my review note for this papaer.

## Summary
* CNN trained on top of pre-trained word vectors
* Pre-trained word vector를 사용한 점은 image에서 pre-trained 모델을 사용한 것에서 착안 (Research Philosophy). 높은 성능향상으로 'universal' feature의 장점을 볼 수 있었음.
   * pre-trained word2vec은 skip-gram이 아니라 CBOW를 사용
   * OOV는 랜덤 초기화 -> OOV를 어떻게 잘 초기화할 것인지를 새로 연구할 필요성이 있다고 함

* task-specific (non-static) vector를 fine-tuning으로 학습해서 성능향상을 보임.
* non-static과 static vector를 가지는 multichannel architecture를 제안
   * multichannel input은 computer vision에서 RGB input과 비슷한 메커니즘
   * multichannel이 overfitting을 완화해주리라 가정했지만, 결과는 데이터에 따라 좋을 때도 있고 나쁠 때도 있었음
   * Multichannel보다 더 regualrize할 수 있는 방법을 향후 연구로 해보자: Non-static vector 전체를 fine-tuning하지 말고 extra dimension을 만들고 여기에만 해보자
* Regularization 테크닉으로 dropout과 L2-norm을 같이 사용

### Reference
Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
