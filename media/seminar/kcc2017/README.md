# 2017 Korea Computer Congress

2017.06.18 ~ 06.20

## 1. 딥러닝의 개념과 응용
- 정교민 교수, 서울대학교
- Backpropagation algorithm에 대하여..  forward와 backward를 반복하면서 학습하는데, 이러한 원리는 EM알고리즘과 비슷하다.

## 2. Convex optimization for decentralized machine learning
- 윤성희 수석연구원, 삼성전자
- convex optimization에서는 최적 해가 항상 존재한다
- 이러한 최적화 background를 알필요가 있는 이유는: 애초부터 (수학적 이론에 의해) 풀리지도 않을 문제를 오랫동안 파라미터 튜닝 등의 헛된 일로 시간을 낭비하지 않을 수 있다. 
- 세상은 non-convex하지만, convex라고 생각하고/변환하고 문제를 해결할 필요가 있다 
- Deep Neural Network는 명백한 non-convex 문제이다
- 반면, SVM은 non-convex한 kernel trick을 쓰더라도 convex 문제이다

#### Why convex optimization?
* many machine learning algorithms (inherently) depend on convex opimization
* one of few optization class that can be actually solved
* a number of engineering and scientific problems can be cast into convex optimization problems
* many more can be apprximated to convex optimization (이런식으로 변환하면 original해는 아니지만 꽤 괜찮은 해를 찾을 수 있다)
* convex optimization sheds lights on intrinsic property and structure of many optimization, hence, machine learning algorithms (ML 알고리즘들이 애초부터 convex optimization이라고 가정되어 설계되었다. 즉, 우리 문제가 convex가 아니더라도 convex인 것 처럼 해석된다)
* 최적화를 통해 (전문성이 필요하여 사람이 해야할 일을) 자동화가 가능케 해준다. 
* 사람이 일일이 최적화할 수 있는 variable의 최대 개수는 대략 5개이다. (사람의 한계) 기계는 100만개까지 풀고 있다.
* 기계학습이 하는 일: what if highly nonlinear and nonconvex fitting function is needed?



#### Optimization examples
* circuit optimization
  - optimization variables: transistor widths, resistances, capacitances, inductances
  - objective: operating speed (or equivalently, maximum delay)
  - constraints: area, power consumption

* machine learning
  - optimization variables: model parameters (e.g. connection weights)
  - objective: squared error (or loss function)
  - constraints: network architecture (fully vs. sparse)
  
#### Solution methods
* for general optimization problems
  - extremely difficult to solve (practically impossible to solve)
  - most methods try to find (good) suboptimal solutions (e.g. using heuristics)
  
* some exceptions
  - least-squares (LS): 18세기 가우스가 사용
  - linear programming (LP): 1940년 2차 세계대전때 사용
  - semidefinite programming (SDP)
  
  
## 3. 딥러닝의 바이오헬스케어 응용
  - 이슬 (한국뉴욕주립대)
  - 특히 바이오 분야에서 전처리하는데 힘을 매우 많이 쏟기 때문에, 딥러닝 방법론이 더 잘쓰이게 되었다. (auto representation learning)
  - CNN: grid type data에 적합. 짤라서 본다. (locality)
  - Restricted Boltzmann Machines: input과 hidden의 에너지를 joint probability로 모델링 (즉, boltzmann으로 모델링)
  - Generative Adversarial Networks: Generator를 이용해 Discrinator를 더 잘 학습하도록 해준다. Gen으로부터 생성된 대표값을 Disc의 입력으로 다시 넣어서 원래 data와 구분되기 어렵도록 Disc를 학습한다.
  - 질 좋은 data도 중요하다. bias된 data이면 true distribution을 representation 하지 못하는 data이다.
  - 멀리있는게 중요하면 RNN을, 가까이에 있는게 중요하면 CNN을..
  
#### Interpretable Deep Learning Models
* 딥러닝은 더이상 블랙박스가 아니다 (해석할 수 있다)
* ICASSP 2017 Tutorial 참고
* Data Generation - Deep generator network proposed by Nguyen et al. 2016  
  

## 4. Deep Learning Toolkit 개발자 입장에서 바라본 DNN이란
  - 송화전 (ETRI)  

  
  
  
  
  
  
