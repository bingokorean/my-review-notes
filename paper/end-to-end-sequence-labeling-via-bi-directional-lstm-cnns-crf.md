# [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf) (2016.6)

# Review note


## Keyword


## Review

---

# Summary note

## Content
* Abstract
* Introduction
* Neural Network Architecture

## Summary
* BLSTM + CNN + CRF
  * state-of-the-art on POS, NER
* End-to-end systems
  * no feature engineering, no data Preprocessing
  * eaily applied to a wide-range of sequential labeling tasks

> dropout을 input layer와 output layer에 모두 넣는다. (다른 논문은 output layer에만 넣는게 더 좋다고 주장하고 있다.)

> POS & NER를 풀기 위한 모델 hyper-parameter set이 동일하다. (learning rate만 빼고) Multi-task learning approach가 된다.

> Embedding의 효과가 POS보다 NER에 월등히 뛰어나다. 이유는 word embedding은 semantic representation이다 하지만 모양은 보지 못한다. POS의 경우 모양을 파악하는게 더 중요하다고 생각할 수 있다.

> CRF 성능을 Out-of-vocabulary test 방식으로 확인하였다. 즉, 양옆 dependency를 이용해 CRF는 사전에 없는 새로운 단어에 대해서도 잘 예측할 수 있다.

## Reference
Ma, Xuezhe, and Eduard Hovy. "End-to-end sequence labeling via bi-directional lstm-cnns-crf." arXiv preprint arXiv:1603.01354 (2016).
