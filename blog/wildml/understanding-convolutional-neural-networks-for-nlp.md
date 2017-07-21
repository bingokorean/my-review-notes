# Understanding Convolutional Neural Networks for NLP

WILDML <br>
Denny Britz <br>
Nov 7, 2015

## Review
The review note for this post is [**here**](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!262&ithint=file%2cdocx&app=Word&authkey=!AMiCI19UuQcRrVM)

## Summary
CNN모델의 특징에 대해 다시한번 정리할 수 있었고 또한 CNN모델과 NLP 특징에 대해서 서로 결부지어서 잘 설명해주었다. 무엇보다 한눈에 CNN for NLP 연구들을 살펴볼 수 있어서 좋았다. 결국 representation 싸움인 듯한데, 도메인(e.g., NLP) 특성에 맞게, 모델(e.g.,CNN) 특성(e.g., 채널개수 활용)에 맞게 (표상법을) 어떻게 잘 설계하느냐에 따라 모델성능에 큰 영향을 준다고 생각한다. 인상깊은 점은 저자가 어떤 개념을 다른 모델/표상법과 비교하면서 (이럴 땐, 무슨 모델과 비슷한 행동한다는 둥) 설명하는 점이었다. (통찰력/내공이 있는듯 하다)
* 왜 Convolution? Location Invariance, Local Compositionality, Sparsitiy Modeling
* 왜 Pooling? Fixed size output, Reduce dim, Detect specific features (bag of n-grams유도), invariance(=locality와 비슷) 
* CNN장점: 다양한 channel들을 통한 representation의 풍부화
* In NLP, the width of the filters = the width of the input matrix
* Image 특징을 이용한 CNN의 철학(locality, etc)이 NLP에도 적용된다. 마치 Bag of n-gram 모델 처럼. (Bag of n-gram처럼 동작하지만, CNN은 심지어 빠르다.)
* CNN에서 stride size를 늘릴수록 Recursive Neural Network과 비슷해진다 (그룹핑을 하면서 상위로 올라가는 느낌을 준다.)
* CNN이지만 relative positions of words를 feature로 사용해서 마치 RNN처럼 동작유도 (denpendency를 많이 요구하는 relation extraction에 CNN 적용)
* Character level의 CNN은 많은 데이터를 필요
