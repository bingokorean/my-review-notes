# [Visualizing and Understanding Recurrent Networks](https://arxiv.org/pdf/1506.02078.pdf) (2015.11)

The review note for this paper is [**HERE(.ppt)**]()

## Summary
* Character-level language model를 interpretable testbed로 하고, long-range dependencies (ex. line lengths, quotes, brackets)를 keep track하는 LSTM의 interpretable cells의 값을 직접적으로 보여주면서 분석한다.
* RNN -> LSTM, GRU로 진화되는 과정을 보면, 모델의 일반성을 높여주는 작업이라고 생각할 수 있다. RNN은 단순히 과거 신호를 받을 수만 있다고 하면, LSTM은 과거 신호를 받고, 수정할 수 있도록 모델이 설계되었다. 즉, 일반성을 더 높여주었다. 물론, 여기에 따른 파라미터 개수를 늘려야되는 단점이 있다. (-> 데이터량 증가 필요)
* RNN, LSTM, GRU를 구분하는 잣대는 coupling form에 있다. 어떻게 coupling form을 수식으로 설계하느냐에 따라 모델이 달라진다. 즉, 한 단계 아래의 input과 이전 시간의 input을 어떻게 결합시키느냐에 따라 모델이 달라진다. RNN은 가장 간단하게 결합하고, LSTM은 gating mechanism을 통해 결합하고, GRU는 functional form으로 결함한다.

_Experiments_
* 본 연구의 Optimization 정책상 network는 100 time step까지만 unroll하고 파라미터를 업데이트한다. 하지만, 최종 학습된 네트워크는 그 단위보다 더 긴 sequence까지도 일반화시킬 수 있다. (즉, LSTM이 longer sequence에 대해 approximately하게 generalize를 잘 한다고 볼 수 있다.)
* Layer2, 3와 다르게 Layer1은 어느 곳으로도 saturated 잘 되지 않았다. 아마 학습이 덜 된 것이 아닐까? Layer1이 output에서 가장 멀리 떨어져있으므로.., 이와 관련해서.. Layer1은 정적 slow하고 layer3는 동적 fast하다. 이것은 준 타니 교수님 모델 MTRNN과 비슷하다.  
* 기본적으로 LSTM은 Long-range dependency를 잘하고, n-gram, n-NN은 short-range dependency를 잘한다. 그래도 이들도 long-term dependency를 잘 할 수 있는데 단, 모델 capacity가 비약적으로 커지는 단점이 있다. 반면, LSTM은 작은 사이즈의 모델 capacity만으로도 long-range dependency를 가질 수 있다. (물론 n이 크면 데이터량이 적어 overfitting이 쉽게 될 우려가 있음)
* LSTM이 n-gram보다 long-rage dependency를 측정하는 task에 대해 성능이 더 좋을 순 있지만, (비율은 작을지라도) LSTM이 못하는 것을 n-gram이 할 수 있는 경우도 있다. 즉, LSTM이 n-gram한테 배울 건 배워야한다. n-gram oracle이 18%나 되는데, 이처럼 LSTM이 모든 면에서 n-gram 모델보다 우수하다고 볼 순 없다.
* LSTM 모델을 scaling up하면, n-gram error를 많이 줄일 수 있다. (위에서 에러 차이에서 n-gram이 차지하는 비율이 매우 높다.) 즉, 모델을 scaling up하면 short-range dependency를 더 잘할 수 있다. 즉, LSTM의 depth, width를 적절히 조절을 잘 해야 long-term dependencies들을 잘 해결할 수 있을 것이다.
* 무엇보다, 이렇게 LSTM 에러를 categorize하고 이해함으로써 LSTM의 약점을 파악할 수 있어 새로운 LSTM architecture를 설계하는데 도움을 준다.
