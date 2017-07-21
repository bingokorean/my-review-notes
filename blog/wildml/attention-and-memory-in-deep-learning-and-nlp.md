# Attention and Memory in Deep Learning and NLP

WILDML <br>
Denny Britz <br>
Jan 3, 2016

## Review
The review note for this post is [**here**](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!261&ithint=file%2cdocx&app=Word&authkey=!AGWbqV-wgtjk4gs)

## Summary
NMT에서의 attention 메카니즘을 설명하고 있다. attention이 memory 메카니즘과 다름없다고 설명하였고 나도 동의한다. RNN에서LSTM진화과정, seq2seq에서attention seq2seq진화과정과 같이 학습할 가중치 개수를 내부 구조적으로 늘려 모델학습이 조금 더 상세히/구체적으로 될 수 있는 '여지'를 만들어주는게 attention이나 memory 메카니즘의 핵심이 아닐까 생각해본다.
* MT에서 정교한 feature engineering을 하는 고전 모델과 다르게 NMT는 하지 않는다. (end-to-end 방법으로 전형적인 딥러닝 철학을 가진다.)
* Seq2seq의 encoder의 last hidden state(=node)로부터 sentence embedding 추출가능
* 긴 한 문장을 하나의 vector로 압축하기엔 Long-dependency 문제: LSTM사용(기본), Reversing input sentence -> 일본-영어 같은 경우 reversing하면 오히려 독 -> attention 메커니즘 등장 
* attention: source sentence의 각 단어에 연결고리를 추가해서 가중치를 단어마다 줄 수있도록 설계, 다른 측면으로 보면 모델을 좀 더 복잡하게 만들어 학습을 좀 더 구체적으로 할 수 있는 여지를 만들어줌
* 인간의 attention(효율성)과 컴퓨터의 attention(복잡성)은 큰 차이. 여기서 attention은 (모든 것들을 기억해야되는) memory 메커니즘과 같다.
* LSTM은 explicit memory, RNN의 hidden node들은 internal memory라 생각할 수 있다.
