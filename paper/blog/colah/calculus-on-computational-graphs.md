# Calculus on Computational Graphs: Backpropagation

Posted on August 31, 2015 <br>
Colah’s Blog


## Review
The review note for this post is [**here**](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!217&ithint=file%2cdocx&app=Word&authkey=!AG7WcdO_nIRAybg).

## Summary
* 어떤 수학 모델은 보통 직렬적 또는 병렬적인 함수들의 집합으로 표현된다. 그리고 그래프 표현법을 사용하면 이러한 함수적 수학 모델을 한눈에 표현하기 쉽다. 그리고 특히, 노드(변수) 간의 영향력을 측정하기 위해 편미분을 사용하는데, 편미분은 그래프의 엣지에서 일어난다고 볼 수 있다.
* 이렇게 기하급수적으로 늘어나는 path때문에 옛날에는 (backpro가 등장하기 전) training이 매우 어려웠다. 원래 모든 경로의 path들을 다 표현하고 이들 모두 더해줘야 하지만, path들을 노드별로 factor만 해버리면 path들을 node기준으로 group으로 묶어 나타내고 곱하기만 하면 된다. (편미분끼리 곱해주는 chain rule과 관련)
* 일반적인 graph구조는 input node는 매우 많고, output node는 하나로 구성된다. 따라서 reverse가 훨씬 효율적이다. 만약 graph가 희한하게 output node가 매우 많고 input node가 하나 또는 매우 적으면 forward가 훨씬 효율적이게 된다. Neural Network 구조는 일반적인 graph 구조를 가지므로 reverse-mode를 사용한다.
