# Probabilistic Topic Models

by M.Blei <br>
Review note for this article is in [here](https://1drv.ms/w/s!AllPqyV9kKUruQyU_x8Pt6tv3Sdc).

## Summary

* (동기) 단순히 문서 자체만을 검색 또는 탐색하는 것이 아닌 문서 속에 스며들어 있는 토픽들을 살펴볼 수 있지 않을까?
* (직관) 문서는 다수의 토픽들을 가지고 있다
   * 사람이 직접 한 문서를 보고 각 주제에 맞게 단어를 색칠할 수 있을 것이다. 완료 후 색칠이 많이 된 특정 주제를 선정할 수도 있을 것이다.
   * 이러한 일련의 과정을 statistical LDA model이 학습 데이터와 함께 자동적으로 할 수 있다.
* (그래피컬 확률 모델) 
   * 랜덤 변수들을 정의하고 이들 모두를 포함하는 결합 확률에서부터 모델의 정의는 시작한다
   * 랜덤 변수들 사이의 의존도(dependency)를 정의하고 이를 모델에 적용시키면 비로소 LDA 모델을 정의할 수 있다
   * 그래피컬 확률 모델 관점으로 보았을 때, 이러한 의존도라는 통계적 가정을 모델에 적용함으로써 결합 확률을 훨씬 더 쉽고 의미있게 구할 수 있도록 해준다
* (추론) 가장 어려운 단계임; 어떻게 보면 LDA 모델의 핵심 알고리즘임; posterior 확률을 구하는 작업; 특히 marginal 확률(=evidence)을 구하는 것이 거의 불가능함
   * posterior 확률을 잘 구하기 위한 연구가 현대적인 베이지안 통계의 관심 분야이다
   * 보통 무식하게 모든 경우를 적분하면서 posterior 확률을 구할 순 없고, 다음 두 가지의 툴을 사용하여 근사치로 구한다
      * sampling based algorithms
      * variational methods
* (가정) 사실, LDA는 강력한 통계적 가정이 포함되어 있음
   * bag of word 가정 : 단어들의 순서 정보를 무시 -> 보완 -> phrase LDA
   * bag of document 가정 : 문서들의 순서 정보를 무시 -> 보완 -> dynamic LDA
   * topic 개수를 미리 정의해야 함 -> 보완 -> bayesian non-parametric LDA
* (기타 등등)
   * LDA 모델의 한 가지 장점은 모듈러하다는 점인데, inference algorithm을 조금만 수정하면 topic parameter와 data-generating distribution을 다른 종류의 데이터에 적용할 수 있다
   
