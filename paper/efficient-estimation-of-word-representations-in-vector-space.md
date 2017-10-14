# Efficient Estimation of Word Representations in Vector Space

My note for this paper (written korean) is in [**HERE (.ppt)**]()

## Summary
* continuous vector representation of words (줄여서 word vector)를 좀 더 빠르고 정확하게 만들 수 있는 모델을 제시
   * 이전 모델들에 비해 간단한 모델 (hidden layer를 없앰)
   * NNLM보다 최대 대략 700배 빨리 학습시킬 수 있음 (물론 data 크기와 word vector dimension 크기에 따라 달라짐)
* 학습된 word vector에 syntactic/semanctic 정보가 있음을 test로 증명 그리고 이 척도로 모델간의 성능을 비교
   * Skip-gram이 CBOW보다 semantic 정보를 훨씬 더 잘 가지고 있음.
   * Skip-gram과 CBOW가 다른 NNLM, RNNLM 모델보다 성능이 더 좋음과 동시에 빨리 학습시킬 수 있음. (물론 빨리 학습시킬 수 있기에 더 많은 data를 사용할 수 있고 이 때문에 성능이 좋아졌다고 볼 수 있음)
* 어떤 challenge에서 Skip-gram + RNNLMs가 best 성능이 나왔음. 이는 pretrained word vector와 neural network model을 같이 사용하는게 좋다고 하는 시사점이 될 수 있음.
