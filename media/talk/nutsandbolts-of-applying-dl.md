# Nuts and Bolts of Applying Deep Learning

Deep Learning School on September 24/25, 2016 <br>
Andrew Ng

My note for this talk is in [**HERE (.ppt)**](https://1drv.ms/p/s!AllPqyV9kKUrhCT4XrOO-WnKuE_N)

## Summary
* 왜 지금 딥러닝인가: (1) scale (2) end-to-end
* 연구자의 능력은 어떤 문제에 봉착했을 때 어떤 decision을 내릴지를 잘 아는 것이다. 
   * workflow를 만들어 상황에 맞게 올바른 decision을 내릴 수 있도록 한다.
* 딥러닝의 장점은 어떤 step에서 stuck하든지 간에 항상 최소한의 한 가지 이상의 action을 취할 수 있다. 
   *예전 모델들은 항상 trade-off에 입각해서 문제를 해결하려 해서 성능을 높이는데 힘이 많이 들었다. 반면 딥러닝 모델은 선택할 수 있는 대안이 많고 좋은 기술을 가지는 툴들이 많다.
* bias/variance를 딥러닝에 맞게 재정의해서 workflow에서 좋은 decision을 내릴 수 있도록 한다. 
   * train-dev set이라는 data를 새로 정의한다
   * human-level performance를 도입하여 bias/variance 문제를 더 잘 풀 수 있도록하였다.
* dev와 test data의 distribution은 같아야 한다


