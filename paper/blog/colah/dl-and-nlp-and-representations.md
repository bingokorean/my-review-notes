# Deep Learning and NLP and Representations
Colah Blog (July 7, 2014)

## [Review-note](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!218&ithint=file%2cdocx&app=Word&authkey=!ACzfCixjpw2Jcmk)

## Summary
Neural Network의 장점을 representation 측면으로 설명하고 있다. NLP의 좋은 representation의 예인 word2vec과 recursive neural network를 설명하고 있다. 특히 word embedding의 특징(similarity, distance)을 강조하면서... 다시 한번 representation의 힘을 느낄 수 있는 계기가 되었고, 모델이 발전하기 위한 목표는 좋은 representation을 만들기 위함인 것 같다.
* hidden layer 1개인 Neural Network는 이론상 hidden node가 충분할시 어떠한 함수를 approximation할 수 있는 universality 특징을 가지지만, 단지 lookup table (규칙기반 모델)과 같은 느낌이라 새로운 데이터에 robust하지 못하다. (즉, NN의 특징(힘)이 되지 못한다.)
* 그냥 5-gram이 valid한지 안한지만 모델링했을뿐인데, 최적화(학습)후에 (weight들을 통해) 단어간의 유사성을 측정할 수 있고, 다음단어를 예측하는 Language Modeling도 할 수 있다. 이것이 바로 자동적으로 좋은 data representation을 배우는 Neural Net의 힘이다.
* 특이한 점은 학습된 "female"vs"male" 그리고 "woman"vs"man" 같은 대립 단어들의 word embedding vector들은 서로 일정한 거리를 가진다. woman자리에 man을 대체하면 문법 오류가 발생하기 때문에 비슷한 벡터가 될 수 없다. (여기서 대립은 단지 하나의 관계일뿐, 다양한 관계를 가진 벡터들도 똑같은 현상이 일어난다. 그리고 이미지 데이터도 똑같이 적용된다.)
* Shared Representation (e.g., bilingual word-embedding, image-word embedding (representation을 잘하기 위한 방법은 무궁무진한 듯 하다.)
* Recursive NN: input 크기가 임의의 길이여도 상관없다. 즉, 계층구조를 가진 tree structure를 input으로 한다. 그리고 reversible sentence representation을 만드는게 특징이다.
