# Deep Learning for Natural Language Processing

Jason Brownlee <br>
MACHINE LEARNING MASTERY

## Contents

* Part1. Introductions
* Part2. Foundations [[**Note**](https://1drv.ms/w/s!AllPqyV9kKUrwAIs7ECAalYKx7Ps)]
* Part3. Data Preparation
   * Clean Text / Vectorize Text
* Part4. Bag-of-Words 
* Part5. Word Embeddings [[**.ipynb**](https://nbviewer.jupyter.org/github/gritmind/review-code/blob/master/blog/deep-learning-for-nlp/contents/word-embedding.ipynb)]
   * Develop, Visualize, Load Word Embedding (skip-gram/cbow, fasttext) / Example of using pretrained glove embedding / Keras Layer Output
* Part6. Text Classification
   * Word Embedding + CNN, Character-level CNN, Deep CNN for Document Classification
* Part7. Language Modeling
   * Charater-, Word-based Neural Langauge Model
* Part8. Image Captioning
   * Encoder-Decoder Model, Pre-trained Model, BLUE
* Part9. Machine Translation
   * Encoder-Decoder Model 

## Projects
* (Part6) CNN models for Sentiment Analysis [[**.ipynb**](https://nbviewer.jupyter.org/github/gritmind/review-code/blob/master/blog/deep-learning-for-nlp/projects/polarity-classification.ipynb)]
   * Develop and Embedding + CNN Model for Sentiment Analysis
   * Develop an n-gram CNN Model for Sentiment Analysis
* (Part7) Develop a Neural Language Model for Text Generation
* (Part8) Develop a Neural Image Caption Generation Model
* (Part9) Develop a Neural Machine Translation Model

<br>


# Part1. Introductions



<br>

# Part2. Foundations

Review note for this part is described in [**here**](https://1drv.ms/w/s!AllPqyV9kKUrwAIs7ECAalYKx7Ps).

## Summary

_Natural Language Processing_

* NLP는 50년 넘은 전통의 언어학-Linguistics(과학)와 컴퓨터(공학)의 합작품이다
* (NLP의 어려움) 사람은 자연어의 정교하고 미묘한 차이를 표현, 인지, 해석하는 데 특출나지만, 이러한 자연어를 수학적인 언어 또는 fomal한 형태로 정의하는 데 잘 못한다
* 언어학(linguistics)은 언어의 규칙(e.g. grammar, semantics, phonetics, ..)을 정의하고 평가하는 일을 함
   * syntax와 semantics를 위한 formal method를 만들었찌만, clean한 mathematical formalisms은 아직까지 해결하지 못함
* 자연어를 이해하는 것은 morphology, syntax, semantics 그리고 pragmatics를 이해하고 인코딩하여 견고한 언어 시스템을 만드는 것임
* 오늘날의 언어학은 computer science를 툴로 사용하는 computational linguisticcs임. (이론적 언어학자가 제안한 문법들을 컴퓨터를 이용해 테스트할 수 있음)
* 많은 데이터와 빠른 컴퓨터의 등장으로, classical top-down rule-based 방법론들은 statistical 기계학습 방법론에 대체되었음
   * 이로 인해, hand-crafted 규칙들에 전적으로 의존하지 않아도 됨 
* Computational linguistics에다가 engineer-based와 statistical method의 empirical 측면을 반영하면, 이를 Natural Language Processing(NLP)이라고 불리기도 함
   * 전통적인 computational linguistic method와 구분 짓기 위해서 statistical NLP라고도 불림
* 우리는 세상을 정교하고 formal하게 표현하지 못하고, 훨씬 더 추상화된 형태로 표현한다. 이러한 측면으로 봤을 때 불확실성이 크다라고 볼 수 있고, statistical 방법론이 이를 잘 모델링할 수 있다
* Statistical NLP는 특히 statistical inference를 하는 특징을 가짐 (i.e. unknown 확률 분포로부터 얻어진 샘플로부터 그 분포를 추론하는 inference)
* 최근 딥러닝의 등장으로 statistical NLP는 새로운 국면에 들어서고 있음

_Deep Learning_

* 딥러닝의 혜택은 scalability와 automatic feature extraction임
* manual로 디자인된 feature는 over-specified되거나 incomplete하기 쉬움
* autmatic feature learning의 핵심은 hierarchy임; lower level feature들이 합성(composition)되어 high level feature가 됨
* end-to-end의 패러다임 변화; sub-system들의 에러가 propagate되는 것을 방지