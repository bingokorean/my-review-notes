# Probabilistic Topic Models

* Materials are mostly from http://www.cs.columbia.edu/~blei
* 내용은 LDA 기초적인 내용에 한정함. (variational inference, advanced LDA 등은 제외)
* Here is [my review note](https://1drv.ms/w/s!AllPqyV9kKUrpULE09mCeSPspRRy)

## Summary
* 문서를 자동적으로 생성하는 생성 모델
* 내부적으로 (문서에 대한) 토픽 분포, (토픽에 대한) 단어 분포를 통해 최종적으로 단어가 생성됨
* 생성 모델이기 때문에 hidden structure의 모든 확률을 추론하게 되면 의미있는 정보들을 얻을 수 있음
* Dirichlet 분포를 prior로 설정하여 분포에 sparsity 능력을 함유시켜 학습과 해석이 더 잘되도록 해줌 (기존 연구와의 차이점)
* 깁스 샘플링을 통해 근사적으로 모델의 추론(or 학습)을 할 수 있음 (원래는 posterior 확률을 구해야 하지만 이는 현실적으로 매우 힘듦)
* 반복적으로 topic assignment를 샘플링하면서 샘플링 방정식을 통해 posterior 확률을 근사적으로 구함

