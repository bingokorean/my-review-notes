# 모두의 머신러닝/딥러닝 - 시즌 강화학습

김성훈 <br>
2017.06.01

## Note
My note for this seminar is in [**HERE (.ppt)**](https://1drv.ms/p/s!AllPqyV9kKUrgm0lFq13AmTUuL5X)

## Summary

**강화학습 구성**

강화학습 구성은 환경 (environment), 보상 (reward; implicit teacher), 행동 (action)으로 이뤄져 있다. 환경은 discrete world라 가정한다 (time-step이 매우 촘촘하면 continuous world와 비슷해진다)
* 환경 (environment): 시간적/공간적 측면 모두 현재의 state가 모든 것을 설명할 수 있나?
   * Partially observable (RNN필요)
   * Fully observable, Markov Decision Proces (FNN필요)
* 보상 (reward): 매 순간 행동에 대해서 피드백이 바로 오지 않고, 게임이 끝나거나 특수한 상황(ex. 벽에 부딪힘)일 때만 비로소 피드백이 온다. (delayed & sparse reward)
* 행동 (action): 강화학습의 목표는 매 state마다 어떤 행동을 취할 것인가 (to find the optimal policy)
   * value-based reinforcement learning (Q-Learning): 각 action에 대응되는 value들을 먼저 찾자; Q함수만 찾으면 optimal policy를 자연스럽게 찾을 수 있다.

**Q함수 구성**

Q함수(Q-table)를 어떻게 정의할까? 환경과 보상의 특징 (보상을 매 시간마다 받을 수 없다는 점, Partially observable하다는 점)을 반영한다. Q-table에 있는 모든 값들을 update하기 위해 기본적으로 여러 번의 실험(episode)가 필요하다. 결국 하나의 path를 찾을 수 있을텐데, 그게 optimal path라는 걸 확신할 수 있을까?
* Q함수 (Q-table)
   * Exploit vs. Exploration: E-greedy (처음에 어영부영 하다 생긴 path만 계속 사용하지 않고 (작은 확률이라도) 새로운 path를 찾아 나서야 한다.
   * Discounted future reward를 해준다. (최단거리로 reward를 받을 수 있는 곳으로 가자)
   * 단, 여기서 Q함수가 true Q함수가 되기 위해서 deterministic world와 finite states 조건을 갖춰야 한다. (stochastic world라면 조금 다르게 Q함수를 바꾼다)

**DQN의 등장**

작은 미로게임 같은 경우는 Q-table크기가 그렇게 크지는 않지만, 엄청나게 큰 게임은 Q-table이 무지하게 클 것이다. 또한 크기와 함께 색깔이 들어가는 게임은 기하급수적으로 Q-table이 커질 것이다. 이런 경우 Q-table을 어떻게 표현하나?
* Q-Network (Q-function Approximation): table처럼 1:1로 매핑하지 않고 네크워크를 사용해 projection시키자. 좀 더 좋은 네트워크를 만들기 위해 Deep Q-Network를 사용한다.
* 단, Q-Network는 Q-table을 approximation한 것이기 때문에 학습을 잘 해야 한다 (converge해야 함). Q-Network converge를 방해하는 (강화학습의 근본적인 문제인) 2가지 를 해결해야 한다. (강화학습에서 지도학습으로 풀기 위해서 다음과 같은 (강화학습에서만 나타나는) 문제점들을 해결해야 한다)
   * Correlations between samples
      * Experience Replay 트릭 사용: 매번 Q-Net을 학습하지 않고 메모리에 일정 개수만큼 저장해놓은 다음, 랜덤하게 가져와서 학습한다.
   * Non-stationary targets
      * Separate Networks: 하나 더 복사한 network의 target값을 C step 동안 사용한다. 




