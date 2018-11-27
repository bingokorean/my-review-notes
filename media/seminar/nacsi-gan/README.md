# NACSI GAN

2017.4.29 <br>
서울대학교 제2공학관 <br>
한국인지과학산업협회 (NACSI), 6시간

## Contents
* [GAN 딥러닝 튜토리얼](https://1drv.ms/w/s!AllPqyV9kKUrgkcOS2UTqI4d3hFc)

## Summary
* 다른 Generator 모델들은 오류를 계산할 때 데이터 간의 직접적인 비교를 했지만, GAN은 discriminator를 사용해서 비교를 했다.
* Generator를 학습할 때는 p(real | G(z))를 max하는 쪽으로 하고, discriminator를 학습할 때는 p(real | x)를 max, p(real | G(z))를 min하는 쪽으로 한다.
* 한 방향으로 최적화를 수행하기 위해 다시 식을 쓰면: Generator를 학습할 때는 (1-p(real | G(z)))를 min하는 쪽으로 하고, discriminator를 학습할 때는 p(real | x)와 (1-p(real | G(z)))를 max하는 쪽으로 한다. 
* Objective function의 확률분포의 expectation을 구하기 위해서는 무한대의 데이터가 있어야 하나 이는 불가능하므로 approximated하게 Monte Carlo 샘플링을 통해 구한다 (사실 이는 자연스럽게 학습데이터를 m개를 사용했을 경우와 같은 뜻이다)
