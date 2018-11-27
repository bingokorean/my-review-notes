# Deep Learning and Dynamic Neural Network Models

## Content
- Mostly based on the book, **EXPLORING ROBOTIC MINDS (Jun Tani)**
- [My review-notes of this book](https://github.com/gritmind/review-paper/tree/master/book/exploring-robotic-minds)

## Project 
OpenAI Gym is a playground to use reinforcement learning. We did it on classic control, catpole and physical movement, bipedal-walker and see how the agents are doing well after training. DQN and DDPG are used for control algorithms. We conducted whether or not the control algorithms are trained well and which parameter set is good to cartpole environment. 
[[Presentation]](https://1drv.ms/p/s!AllPqyV9kKUrgmWCZjce7es5EFbB) [[Report]](https://1drv.ms/w/s!AllPqyV9kKUrgmRNnYiZ_rP31VaQ) [[Code]](https://github.com/gritmind/review-media/tree/master/class/deep-learning-and-dynamic-neural-network-models/project)

**OpenAI Gym을 활용한 강화학습** <br>
각 게임에 적합한 딥러닝 모델과 파라미터셋 찾으면 최적화가 가능하게 되는지 또는 빨리될 수 있는지 알아보자.

파라미터셋
  * FNN(fully) vs. RNN(patially observable) - (FNN만 사용)
  * DQN(discrete) vs. DDPG(continuous action))
  * game over reward size 
  * experience replay size
  * with or without separate networks

일반적인 기계학습 모델을 사용하는 문제와 같이 절대적인 기준은 없고 특정 데이터 또는 특정 게임마다 파라미터셋이 달라진다.
Cartpole 게임에서는 experience replace size가 클수록 separate target network는 하지 않을수록 좋았고, game over reward는 100이 가장 좋았다.
파라미터셋이 좋지 않으면 매우 느리게 수렴하거나 심지어 수렴하지 않는 경우도 있다.
따라서, 적합한 파라미터셋을 찾는 것이 강화학습에서는 매우 중요하다.
  
## Acknowledgement
[EE817] Deep Learning and Dynamic Neural Network Models, Jun Tani, Kaist
