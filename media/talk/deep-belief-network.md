# Deep Belief Network

BIGDATA ACADEMY <br>
서울대학교, 원중호

## Review
Review note is [**HERE**](https://1drv.ms/w/s!AllPqyV9kKUrgza8NuXM5B6wIZaK)

## Summary
* 생성 모델은 데이터가 태생부터 어떻게 생성되는지까지의 과정을 수학적으로 모델링 한 것
* 생성 모델은 비지도 학습 및 준지도 학습이 가능한 장점이 있음
* DBN은 여러 개의 RBM 기반 생성 모델로서 훈련 방법은 다음과 같은 두 단계로 구성됨
   * pre-training - 각 층마다 순차적으로 실행 (비지도 학습)
   * fine-tuning - 전체 네트워크를 역전파 알고리즘으로 학습 (지도 학습)
* RBM 모델 구조
   * 양분그래프(bipartite graph) - 입력과 출력을 서로 뒤바꿀 수 있음
   * 입/출력 관계를 확률적인 모형인 에너지 함수로 표현 가능
   * 에너지 함수에는 입/출력에 각각 일차원 선형관계, 입/출력 모두에는 이차원 선형관계를 가짐
   * 에너지 함수에 sigmoid 함수를 사용하여 마치 신경망 처럼 구성될 수 있도록 함
* 이러한 RBM을 여러 개 쌓으면 stacked autoencoder과 deep belief network (DBN)을 만들 수 있음
* 생성 모델을 학습하기 위해 가능도(likelihood)를 미적해서 얻은 gradient를 사용; Gibbs sampling으로 근사적으로 gradient를 구함.
