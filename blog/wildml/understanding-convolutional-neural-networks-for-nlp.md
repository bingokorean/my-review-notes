# Understanding Convolutional Neural Networks for NLP

WILDML <br>
Denny Britz <br>
Nov 7, 2015

## Review
The review note for this post is [**here**](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!262&ithint=file%2cdocx&app=Word&authkey=!AMiCI19UuQcRrVM)

## Summary
* 왜 Convolution? Location Invariance, Local Compositionality, Sparsitiy Modeling
* 왜 Pooling? Fixed size output, Reduce dim, Detect specific features (bag of n-grams유도), invariance(=locality와 비슷) 
* CNN장점: 다양한 channel들을 통한 representation의 풍부화
* In NLP, the width of the filters = the width of the input matrix
* Image 특징을 이용한 CNN의 철학(locality, etc)이 NLP에도 적용된다. 마치 Bag of n-gram 모델 처럼. (Bag of n-gram처럼 동작하지만, CNN은 심지어 빠르다)
* CNN에서 stride size를 늘릴수록 Recursive Neural Network과 비슷해진다 (그룹핑을 하면서 상위로 올라가는 느낌을 준다.)
* CNN이지만 relative positions of words를 feature로 사용해서 마치 RNN처럼 동작유도 (denpendency를 많이 요구하는 relation extraction에 CNN 적용)
* Character level의 CNN은 많은 데이터를 필요
