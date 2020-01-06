# Colah's Blog Review

<br>

## Calculus on Computational Graphs: Backpropagation

* August 31, 2015

### Review
The review note for this post is [**here**](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!217&ithint=file%2cdocx&app=Word&authkey=!AG7WcdO_nIRAybg).

### Summary

* 어떤 수학 모델은 보통 직렬적 또는 병렬적인 함수들의 집합으로 표현된다. 그리고 그래프 표현법을 사용하면 이러한 함수적 수학 모델을 한눈에 표현하기 쉽다. 그리고 특히, 노드(변수) 간의 영향력을 측정하기 위해 편미분을 사용하는데, 편미분은 그래프의 엣지에서 일어난다고 볼 수 있다.
* 이렇게 기하급수적으로 늘어나는 path때문에 옛날에는 (backpro가 등장하기 전) training이 매우 어려웠다. 원래 모든 경로의 path들을 다 표현하고 이들 모두 더해줘야 하지만, path들을 노드별로 factor만 해버리면 path들을 node기준으로 group으로 묶어 나타내고 곱하기만 하면 된다. (편미분끼리 곱해주는 chain rule과 관련)
* 일반적인 graph구조는 input node는 매우 많고, output node는 하나로 구성된다. 따라서 reverse가 훨씬 효율적이다. 만약 graph가 희한하게 output node가 매우 많고 input node가 하나 또는 매우 적으면 forward가 훨씬 효율적이게 된다. Neural Network 구조는 일반적인 graph 구조를 가지므로 reverse-mode를 사용한다.

<br>

## Understanding LSTM Networks

* Aug 27, 2015
* The review note for this blog is in [**here**](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!282&ithint=file%2cdocx&app=Word&authkey=!ACX3_SzAnuGJmjI)
* Also, the presentation is in [**here**](https://1drv.ms/p/s!AllPqyV9kKUrhFeUY9qtEqflUyP1)

### Summary
LSTM 구조를 자세하게 step-by-step으로 설명하고 있다. LSTM 구조를 이해하는데 읽어야될 필수적인 문서가 아닐까 생각해본다.
* RNN은 loop구조를 통해 정보의 지속성(persistence)을 가질 수 있도록 유도된 모델이다.
* RNN은 Set of FNN라고 생각할 수 있다. 단지, hidden layer들은 모두 연결되어 있다. 따라서, sequence 또는 list 형태로 이뤄진 FNN set이라고 볼 수도 있다. 
* 짧은 문장 대상으로는 RNN이 잘 동작하지만, 긴 문장에서는 Long-Term Dependency 때문에 잘 안된다. (LSTM 등장)
* LSTM은 애초에 구조상으로 history 정보들의 크기를 기억하도록 설계되어 있다. (즉, default가 histroy 정보들을 기억한다. 학습으로 기억하는게 아니라.. 학습하면서는 강도를 조절할 뿐이다.)
* LSTM은 RNN의 generalization version이라고 생각할 수 있다. (RNN 모델이 LSTM 모델에 포함되는....)
* RNN은 하나의 'tanh' layer 구조/모듈에 의해 반복되고, LSTM은 서로 상호작용하는 4개의 구조/모듈에 의해 반복된다.
* LSTM구조의 핵심은 'cell state'이다. 전체를 통과하는 하나의 길다란 connection이 더 생겼다. (마치 conveyor belt와 흡사하다.) connection을 추가해서 모델 capacity를 늘린 것과 마찬가지이다. 
* 모델 Capacity관점으로 보면, RNN은 Capacity가 작다. history 정보들의 크기를 조절할 수 있는 parameter가 없다.
* LSTM은 이러한 cell state를 통해 정보(information)를 remove(throw away, drop) or add(store)한다. 이렇게 정보의 흐름을 통제할 수 있는 gate 구조(sigmoind neural net + pointwise multiplication)를 도입하였다.
* LSTM은 cell state를 바라보면서 정보를 remove (forget gate)하고 (e.g., gender of old subject를 잃어버림), 정보를 store (input gate)하고 (e.g., gender of new subject를 기억함), 정보를 outout (output gate)하기도 한다 (e.g., relevant to a verb를 출력. 동사에 단/복수형 일치하기 위해). 단지, cell state의 정보는 흐르고 있을뿐인데, gate들이 중간에서 filter역할을 해준다. 그리고 이러한 gate는 하나 혹은 여러개의 Neural Network로 이뤄져있다. (Neural Network안에 또 다른 Neural Network들이 존재한다.) 
* 'cell state'의 filter 역할을 하는 gate를 어떻게 구성하느냐에 따라 여러 버젼들의 LSTM이 생긴다. cell state도 input으로 받아들이는 gate구조를 가지는 "peephole connections" gate 구조가 있고, forget gate와 input gate를 결합하고, cell state와 hidden state도 결합하는 GRU도 있다. (이렇게 하나로 통일하면 기존 LSTM보다 파라미터 수를 적게 가지게 되는 장점이 있다. 아마 LSTM 본연의 역할을 유지하면서 최소한의 파라미터 수를 가지도록 설계한 듯 하다.)
* RNN의 진화는 LSTM, 그 다음 진화는 attention model이다. RNN의 모든 step들이 모든 정보들을 받아들이는 것. (결국 connection 문제이다. 길을 열어주느냐 마느냐...)

<br>

## Deep Learning and NLP and Representations

* July 7, 2014

### [Review-note](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!218&ithint=file%2cdocx&app=Word&authkey=!ACzfCixjpw2Jcmk)

### Summary
Neural Network의 장점을 representation 측면으로 설명하고 있다. NLP의 좋은 representation의 예인 word2vec과 recursive neural network를 설명하고 있다. 특히 word embedding의 특징(similarity, distance)을 강조하면서... 다시 한번 representation의 힘을 느낄 수 있는 계기가 되었고, 모델이 발전하기 위한 목표는 좋은 representation을 만들기 위함인 것 같다.
* hidden layer 1개인 Neural Network는 이론상 hidden node가 충분할시 어떠한 함수를 approximation할 수 있는 universality 특징을 가지지만, 단지 lookup table (규칙기반 모델)과 같은 느낌이라 새로운 데이터에 robust하지 못하다. (즉, NN의 특징(힘)이 되지 못한다.)
* 그냥 5-gram이 valid한지 안한지만 모델링했을뿐인데, 최적화(학습)후에 (weight들을 통해) 단어간의 유사성을 측정할 수 있고, 다음단어를 예측하는 Language Modeling도 할 수 있다. 이것이 바로 자동적으로 좋은 data representation을 배우는 Neural Net의 힘이다.
* 특이한 점은 학습된 "female"vs"male" 그리고 "woman"vs"man" 같은 대립 단어들의 word embedding vector들은 서로 일정한 거리를 가진다. woman자리에 man을 대체하면 문법 오류가 발생하기 때문에 비슷한 벡터가 될 수 없다. (여기서 대립은 단지 하나의 관계일뿐, 다양한 관계를 가진 벡터들도 똑같은 현상이 일어난다. 그리고 이미지 데이터도 똑같이 적용된다.)
* Shared Representation (e.g., bilingual word-embedding, image-word embedding (representation을 잘하기 위한 방법은 무궁무진한 듯 하다.)
* Recursive NN: input 크기가 임의의 길이여도 상관없다. 즉, 계층구조를 가진 tree structure를 input으로 한다. 그리고 reversible sentence representation을 만드는게 특징이다.

