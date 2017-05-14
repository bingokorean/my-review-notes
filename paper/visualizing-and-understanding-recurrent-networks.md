# [Visualizing and Understanding Recurrent Networks](https://arxiv.org/pdf/1506.02078.pdf) (2015.11)

# Review note
[[PPT Slides]](https://1drv.ms/p/s!AllPqyV9kKUrgkvfr_bd7b6LKoK-)

## Keyword


## Review

---

# Summary note

## Content
* Abstract
* Introduction
* Related Work
* Experimental Setup
* Experiments
* Conclusion


## Summary

### 0. Abstract
* LSTM을 포함한 RNN 계열 모델들이 sequential data를 해결하는 기계학습 문제에서 성공적인 실험 결과를 보이고 있다.
* 현실에서(in practice) LSTMs이 좋은 성능을 내고 있지만, the source of their performance and their limitations remain rather poorly understood.
* Character-level language model를 interpretable testbed로 하고, LSTM 모델의 성능측면으로 장/단점을 파악해보자. (LSTM의 representations, predictions, error types을 분석함으로써)
  * 특히, 우리 실험은 long-range dependencies (ex. line lengths, quotes, brackets)를 keep track하는 **interpretable cells**의 값을 직접적으로 보여주면서 분석한다.
  * short-range dependency를 잘 catch하는 n-gram모델과 LSTM 모델을 비교하면서, LSTM의 long-range dependency의 성능 향상 측면을 간접적으로 확인한다.
  * We provide analysis of the remaining errors and suggests areas for further study.


### 1. Introduction
* RNNs (specifically LSTM) are effective models in applications that involve sequential data
  * Language modeling (Mikolov et al. 2010)
  * handwriting recognition and generation (Graves et al. 2013)
  * video analysis (Donahue et al. 2015)
  * image captioning (Vinyals et al. 2015; Karpathy & Fei-Fei. 2015)

However, both **the source of their impressive performance** and **their shortcomings** remain poorly understood. <br>
Therefore, **Our goal is..** with the source of their performance + their shortcomings, we can design better architecture.

* (Related work) LSTM 내부구조를 이해하는 관련 연구는 몇 가지 있지만...
  * the effects on performance as various gates and connections are removed (Greff et al. (2015); Chung et al. (2014))
    * -> However, while this analysis illuminates the performance-critical pieces of the architecture, it is still limited to examining the effects **only on the global level** of the final test set perplexity alone.
  * LSTM can store and retrieve information over long time scales using its gating mechanisms (Hochreiter& Schmidhuber (1997))
    * -> However, it is **not immediately clear** that similar mechanisms can be effectively discovered and utilized by these networks **in real-world data**, and with the common use of simple stochastic gradient descent and truncated backpropagation through time.


* 따라서, 우리 연구는 이렇게 진행한다. our work provides the first empirical exploration of **the predictions of LSTMs** and **their learned representations** on real-world data.
  * Contretely, we use **character-level language models** as an interpretable testbed for illuminating the long-range dependencies, high-level patterns (ex. line lengths, brackets, and quotes) ((long-range dependencies가 요구되는) character-level language model을 사용)
  * n-gram model prediction과 LSTM prediction을 비교: we find that LSTMs perform significantly better on characters that require long-range reasoning (short-range dependencies를 잘하는 n-gram model들과의 비교)
  * error analysis with several categories (하나의 LSTM error를 여러 종류의 type별로 쪼개서 분석함으로써 특히 LSTM의 제한점을 파악)

> LSTM의 output을 보고 (단순 성능) 판단할 뿐만 아니라, LSTM의 내부적인 module들의 output (=memory cell)도 보고 판단해서 LSTM의 원리를 좀 더 정확히 이해하고자 한다. 그리고 이를 통해 새로운 아키텍쳐 설계하는데 도움을 주고자 한다.

### 2. Related Work
#### 2.1 Recurrent Networks
* Despite RNNs’ early successes, the difficulty of training naïve recurrent network has encouraged various proposal for improvements where we can store and retrieve information over long time
  * LSTM with explicit gating mechanisms (Hochreiter & Schmidhuber. 1997)
  * GRU with functional forms (Cho et al. 2014)
  * content-based soft attention mechanisms (Bahdanau et al. 2014)
  * push-pop stacks (Joulin & Mikolov. 2015)
  * (more generally) external memory arrays with both content-based and relative addressing mechanisms (Graves et al. 2014)
* In this work we focus the majority of our analysis on the LSTM due to its widespread popularity and a proven track record.

> RNN -> LSTM, GRU로 진화되는 과정을 보면, 모델의 일반성을 높여주는 작업이라고 생각할 수 있다. RNN은 단순히 과거 신호를 받을 수만 있다고 하면, LSTM은 과거 신호를 받고, 수정할 수 있도록 모델이 설계되었다. 즉, 일반성을 더 높여주었다. 물론, 여기에 따른 파라미터 개수를 늘려야되는 단점이 있다. (-> 데이터량 증가 필요)

#### 2.2 Understanding Recurrent Networks
* basic LSTM architecture의 extended 연구는 활발하지만, relatively little attention has been paid to understanding the properties of its representations and predictions.
  * a comprehensive study of LSTM components (Greff et al. 2015)
  * evaluated GRU compared to LSTMs (Chung et al. 2014)
  * an automated architecture search of thousands of RNN architectures (Jozefowicz et al. 2015)
  * the effects of depth (Pascanu et al. 2013)
* These approaches study recurrent network based **only on the variations in the final test set cross entropy**, `while we break down the performance into interpretable categories and study individual error types`.
  * (Most related to our work) the long-term interactions learned by recurrent networks in the context of character-level language models, specifically in the context of parenthesis closing and time-scales analysis (Hermans & Schrauwen. 2013)
    * -> Our work complements their results and provides additional types of analysis
  * (heavily influenced by work on in-depth analysis of errors) the final mean average precision is similarly broken down and studied in detail (object detection, Hoiem et al. 2012)

### 3. Experimental Setup
[[PPT Slides]](https://1drv.ms/p/s!AllPqyV9kKUrgkvfr_bd7b6LKoK-)
* In particular, it was observed that the backpropagation dynamics caused the gradients in an RNN to either **vanish** or **explode**.
  * It was later found that the exploding gradient concern can be alleviated with a heuristic of **clipping** the gradients at some maximum value Pascanu et al. (2012).
  * `On the other hand, **LSTMs** were designed to mitigate the vanishing gradient problem.`

> LSTM은 오로지 RNN의 gradient vanishing problem만 해결하기 위해 등장하였다. exploding문제는 clipping 트릭으로 해결한다. 그렇다면, RNN은 vanishing과 exploding 두 가지 모든 문제를 가지고 있다. LSTM은 vanishing을 그나마 완화시킨다. exploding은 아마 비슷할 듯 하다...

* The precise mathematical form of the recurrence (**coupling form** (= interacting/transforming form) between the inputs from the layer below in depth and before in time) varies from model to model.
  * RNN: coupling form is an **additive interaction** (while LSTM & GRU's coupling form is **multiplicative interactions**)
  * LSTM: At each time step the LSTM can choose to read from, write to, or reset the cell using **explicit gating mechanisms**.
  * GRU: The GRU has the interpretation of computing a candidate hidden vector and then smoothly **interpolating** towards it gated by z.

> RNN, LSTM, GRU를 구분하는 잣대는 coupling form에 있다. 어떻게 coupling form을 수식으로 설계하느냐에 따라 모델이 달라진다. 즉, 한 단계 아래의 input과 이전 시간의 input을 어떻게 결합시키느냐에 따라 모델이 달라진다. RNN은 가장 간단하게 결합하고, LSTM은 gating mechanism을 통해 결합하고, GRU는 functional form으로 결함한다.

* Total Weight Dimension
  * RNN : 2*n*n
  * LSTM : 8*n*n
  * GRU : 6*n*n



### 4. Experiments
[[PPT Slides]](https://1drv.ms/p/s!AllPqyV9kKUrgkvfr_bd7b6LKoK-)

#1
> Output probability vector를 t-SNE을 사용해 2D로 mapping할 수도 있다. (RNN은 고유의 cluster를 가지고, LSTM과 GRU는 비슷한 cluster를 가진다.)

#2
> backpropagation 때문에 LSTM이라도 dependency가 100 character가 넘어가면 gradient signal이 거의 없어지는데, 실험에서 dependency가 ~230이더라도 학습이 되는 것을 확인할 수 있다. (LSTM이 longer sequence에 대해 approximately하게 generalize를 잘 한다고 볼 수 있다.)

#3
> gate들이 right-saturated됐는지 left-saturated됐는지에 따라서 정보의 흐름를 파악할 수 있다. (예를 들어, forget gate가 완전히 right-saturated된다면, remember values for very long time periods, 어느쪽으로 saturated하지 않는다면 그냥 feed-forward한다고 생각할 수 있다.)

#4
> LSTM은 Long-range dependency를 잘하고, baseline(n-gram, n-NN) short-rage dependency를 잘한다. 물론 n이 클 경우 long-range dependency도 잘한다. 하지만, n이 매우 커야되는데, 이는 모델 capacity가 비약적으로 커진다는 문제가 있다. 반면, LSTM은 작은 사이즈의 모델 capacity만으로도 long-range dependency를 가질 수 있다.

#5
> LSTM이 n-gram보다 long-rage dependency를 측정하는 task에 대해 성능이 더 좋을 순 있지만, (비율은 작을지라도) LSTM이 못하는 것을 n-gram이 할 수 있는 경우도 있다. 즉, LSTM이 n-gram한테 배울 건 배워야한다.

#6
> LSTM은 closing brace, carriage return 등 long-range dependency가 필요한 character를 특히나 예측을 잘한다.

#7
> 학습 epoch이 진행될수록 LSTM의 long-range dependency에 대한 성능은 높아진다 (LSTM 특징). 이는 seq2seq 논문에서 input 문장을 reverse하면 성능이 더 잘나왔다는 현상?에 뒷바침해준다. 왜냐면 더 가까이 있는 것들은 더욱 더 잘나오고, 더 멀리 있는 것들은 충분한 학습으로 long-rage dependency를 획득하기 때문이다. The inversion introduces short-term dependencies that the LSTM can model first, and then longer dependencies are learned over time.

#8
> 무엇보다, 이렇게 LSTM 에러를 categorize하고 이해함으로써 LSTM의 약점을 파악할 수 있어 새로운 architecture를 설계하는데 도움을 준다.

#9
> LSTM 모델을 scaling up하면, n-gram error를 많이 줄일 수 있다. (위에서 에러 차이에서 n-gram이 차지하는 비율이 매우 높다.) 즉, 모델을 scaling up하면 short-range dependency를 더 잘할 수 있다.



### 5. Conclusion
* Character-level language models as interpretable testbed
  * For analyzing the predictions, representations training dynamics and error types present in RNNs
* Long-range dependency on real-world data (LSTM 장점 확인)
  * via Visualization experiments and cell activation statistics
  * via Comparisons to finite horizon n-gram models
* Error Type Analysis
  * Study the limitations of LSTMs
* In particular, N-gram Type Error
  * When scaling up the model, it has large portion. (즉, scaling up하면 n-gram이 잘하는 short-range dependency를 잘 할 수 있다.)
  * 이러한 근거가 새로운 아키텍쳐를 설계할 수 있는 힌트가 되어준다. (Some evidence for building new architecture)



## Reference
Karpathy, Andrej, Justin Johnson, and Li Fei-Fei. "Visualizing and understanding recurrent networks." arXiv preprint arXiv:1506.02078 (2015).
