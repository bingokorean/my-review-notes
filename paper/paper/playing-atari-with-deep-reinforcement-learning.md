# Playing Atari with Deep Reinforcement Learning

V.Mnih <br>
DeepMind Technologies

My note for this paper is in [**HERE(.ppt)**](https://1drv.ms/p/s!AllPqyV9kKUrhCa0tfenouJZg_MT)

## Summary
* converge가 어려운 딥러닝 모델(CNN)을 강화학습 알고리즘에 성공적으로 적용시키고 Atari Game에서 좋은 성능을 내고 사람도 이겼음.
* 기존연구와 달리 prior knowledge도 feature engineering도 없는 end-to-end 딥러닝 모델을 action-value function을 디자인하기 위해 사용.
* 딥러닝을 converge를 잘시키기 위해 stochastic gradient descent와 experience repaly 기법을 사용.
   * 이는 강화학습의 문제점인 correlated data와 non-stationary distribution를 완화시켜줌.
* Experience Replay이 좋은 효과를 줬으나 한계점도 이다: memory buffer가 유한하기 때문에 recent transition들만 포함되어 있음. 그리고 uniform sampling을 하므로 중요한 transition을 강조할 수 없음.
* 모델 아키텍쳐와 hyper-parameter를 하나로 고정시키고 여러 종류의 아타리 게임들에게 적용해도 괜찮았음. (reward만 -1, 0, 1로 discrete화 시켜줌)
* 사람보다 성능이 현저히 떨어지는 게임들이 있었는데 이들의 특징은 긴 시간 동안 이뤄지는 전략을 필요로 하는 게임이었음. (현재 DQN 강화학습의 한계점)
