# Going Deeper with Convolutions

A brief summarized document is [**here**](https://1drv.ms/w/s!AllPqyV9kKUrgUpF-MHcOw6A51u4). <br>
A related presentation is [here](https://1drv.ms/p/s!AllPqyV9kKUrgUvgumcHYR4UPOkQ).

## Summary
* 효율적인 이미지 인식을 위한 인셉션 모듈을 활용한 매우 깊은 컨볼루션 신경망 모델 제시
* 7개 모델을 사용한 앙상블 모델
* 깊은 모델이 학습이 잘 되도록 하기 위해 auxiliary classifier들을 중간중간에 추가하였음 (이로 인해 단계적으로 여러 개의 에러 시그날들을 통해 학습함) 

_인셉션 모듈_
* 인셉션(inception) 모듈로 네트워크의 width와 depth를 확장시키고 계산적 효율성을 가짐 (메모리 효율성으로 임베디드 시스템에도 사용될 수 있음)
* 마치 R-CNN에서 object 감지를 먼저 하는 것처럼, 인셉션 모듈이 object 감지를 하는 역할을 함. (딥러닝 모델은 blackbox라 할지라도 내부적으로 단계적으로 해야할 일을 기준으로 모델을 설계함)
* 인셉션 모듈은 1x1, 3x3, 5x5 convolution과 같이 여러 종류를 사용하는데, 이는 multiple scale을 가지는 Gabor filter에서 영감을 얻음 것임.
* 기존 Gabor filter와의 차이점은 인셉션 모듈을 9개 정도 둬서 같은 과정을 여러 번 반복하겠다는 의도가 있음
* 기존 Network-in-Network와의 차이점은 depth를 늘리는 용도뿐만 아니라 1x1을 통해 차원 축소를 통해 parameter와 계산적 비용을 줄이기 위한 용도로도 사용됨
* 지금까지는 connection-level로 sparsity 능력을 많이 부여해서 overfitting를 방지하고자 함. (여러 가지 한계 (e.g. 병렬처리제한))
* 본 연구에서는 인셉션 모듈과 1x1 컨볼루션으로 filter-level로 sparsity 능력을 추가로 부여하고자 하였음.



