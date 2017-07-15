# Understanding LSTM Networks
Colah Blog (Aug 27, 2015)

## [Review-note](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!282&ithint=file%2cdocx&app=Word&authkey=!ACX3_SzAnuGJmjI)

## Summary
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
