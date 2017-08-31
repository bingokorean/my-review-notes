# 프로그래머를 위한 알파고

Donghun Lee (이동헌) <br>
Donghun2014@gmail.com

## Note
The ppt-note for this slideshare is [**HERE**](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!285&ithint=file%2cdocx&app=Word&authkey=!AJbJavozqxoNvnI).

## Summary
알파고의 기초지식을 습득하기에 좋은 자료이다. 모든 내용을 수식과 그리고 그림과 함께 친절하게 설명해주고 있다. 알파고 논문을 보기전에 참고하면 좋을 것 같다.
* 알파고에서 근본 기술은 DL, RL, MCTS로 크게 3가지로 구성되어 있다. 두뇌 역할을 하는 기술은 다음 수를 예측하는 policy network와 판세를 예측하는 value network로 나뉠 수 있다. 기본적인 동작은 (DL, RL을 사용한) policy network와 MCTS를 통해 다음 수를 예측하지만, 경우의 수가 매우 많은 초반의 경우에는 이들을 사용하기 보다는 다음 수가 아닌 곧바로 판세를 예측할 수 있는 (DL을 사용한 (간접적으로 RL사용)) value network를 자주 사용한다. (실제로 MCTS는 Policy와 Value Net를 서로 혼용해서 사용. Policy는 Value보다 정확한 반면, 속도가 느림.) DL은 policy/value network을 구축하는데 사용되고, RL은 DL로 구축된 policy network의 성능을 높이기 위해 사용된다. MCTS는 최적의 경로를 찾는데 사용된다.
* Policy Net의 경우 input은 19x19x48 binary array이고 output은 19x19의 probability distribution (다음수 예측)이고 Value Net의 경우 input은 19x19x49 binary array이고 output은 a single probabilistic value (판세 예측)이다. 
* Value Network의 output layer는 tanh unit으로 구성. 출력값을 [-1, 1]로 매핑시켜 판세를 분석할 수 있도록 해준다.
* 알파고는 실제로 시뮬레이션을 많이 하고, 가장 많이 선택된 착수를 선택한다. 시뮬레이션을 빨리하기 위해서 MCTS가 큰 역할을 해준다.
* data (기보) augmentation을 하기 위해  면대칭, 점대칭 바둑판 구성을 하였다.
* Value Net을 학습하기 위해 RL Policy Net끼리 자체대결을 실시하였다. (자체대결의 결과를 labeled data로 간주.) Overfitting을 줄이기 위해 (자체대결) 한 판당 하나의 position만 data로 사용하였다. (실제로 한판당 평균 200~250 position 존재)
