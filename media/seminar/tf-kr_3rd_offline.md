# Tensorflow-KR 3차 오프라인 모임

2019.10.20


* On-device ML with TensorFlow 2.0 and TFLite (Jason Zaman, Light Labs)
   * https://github.com/perfinion
   * Quantization을 했지만 성능이 크게 떨어지지 않았다 (MobileNet)

* Auto Scalable 한 Deep Learning Production을 위한 AI Service Infra 구성 및 AI DevOps Cycle (김훈동, SKT)
   * Flask, Cloud PaaS, TensorRT, ...
   * Pandas만 빼도 성능이 20% 향상. 큰 데이터 자체를 인자로 넘기는 함수의 사용 횟수를 줄여야 함.
   * fucntiontools.partial, memoryview
   * 병목이 어디서 주로 발생하는지 찾는 것이 중요 - 메모리?, CPU?, IO?
   * Pandas UDF (온라인보다는 배치), Horovod (Spark 위에서 메모리 기반), Petastorm (병렬, 로딩), Horizon (강화학습), Airflow, TFX, MLflow, Rapids.ai, Clipper.ai (캐쉬 적용), ONNX.ai (압축)
   * AI 프로덕트는 비동기에 좋다 (weight 계산만 하니)
   * Kubernetes - 켜져 있는 인프라를 가상화해서 docker로 만든다
  
* 프로덕션 환경에서 연구하기 (하성주, Hyperconnect)
   * 주로 경량 모델을 모바일에 배포하는 일, 실시간 이미지 처리를 다루는 서비스
   * 신기술을 업그레이드해서 모바일에 최적화된 기계학습 기술 개발
   * 키워드 검출 (Keyword Spotting)
   * 목표 - 정확도 + 실시간 레이턴시
   * 개발 프로세스 한 바퀴를 rough하게 먼저 돌리고 병목지점을 파악하고 가능여부를 짐작한다.
   * 오디오 프로세싱 - 오디오 신호를 푸리에 변환을 해서 coefficient 벡터를 이미지화해서 이미지 처리를 할 수 있도록 한다.
   * shallow network에 대한 고민. 다중 채널 1D convolution.
   * 2D를 1D로 바라본다.
   * TC-ResNet
   * ablation text; 논문을 쓰면 좋은 점이 자꾸 실험해보면서 불필요한 모듈을 파악할 수 있다는 점.

* 당근마켓 개인화 추천 시스템 (전무익, 당근마켓)
   * 팀 블로그 - https://medium.com/daangn
   * 피드(feed) 서비스 - 특별한 이유 없이 사람들이 일단 들어올 수 있도록..
   * 유투브 추천 시스템 참고 - 후보 모델만 선택해서 서비스 중
   * 유사 벡터 검색을 빠르게 하기 위해 faiss 라이브러리 활용
   * 빅쿼리(BigQuery)에 데이터 저장
   * SQL 쿼리로 데이터를 필터링해서 학습 데이터를 만듦
   * 학습 데이터는 TFRecord에 저장
   * 병렬 분산 처리 - Tensorflow Transform
   * Cloud AI Platform, Tensorflow Estimator
   * kubeflow pipeline
   
* Kafka 스트림을 위한 멀티프로세스 딥러닝 추론 (이성철, Naver)
   * 대용량 데이터 플랫폼; 메시지 기반 데이터 전송
   * 처음엔 상품명만, 나중에 상품명+이미지 활용, 몰 카테고리 정보도 활용.
   * CNN for Text, CNN for Image, 앙상블
   * multiprocess Process 상속. run함수 수정. -> 카프카 스트림 처리
   * 나누기는 최대한 GPU에서 연산될 수 있도록 해야 함
   * TFRecord로 데이터(Spark)와 모델(TF)을 연결
   * tf.data 모듈

* Graph neural networks (류성옥, KAIST)
   * https://github.com/SeongokRyu
   * Inductive Biases in Neural Network 
   * CNN, RNN => 의도적으로 bias를 유도한다. 여기서 bias는 틀, 형태와 같은 추상적 의미로 생각해도 되겠다. 
   * transformer의 inductive bias는 self attention, 그리고 positional encoding이다.

* 나만의' 코퍼스는 없다? 자연어처리 연구 데이터의 구축, 검증 및 정제에 관하여 (조원익, 서울대학교)
   * 언어화된 데이터는 다른 데이터의 semantic한 측면과 연관될 수 있음
   * 어노테이션의 종류 - 통사(syntax)에서 의미(semantics)를 넘어 화용(pragmatics)까지
   * 사실 영역은 종종 그 경계가 흐려진다 
   * The Life Cycle of a Linguistics Question
   
<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/life_cycle_linguistics.PNG" width="80%" height="80%"></p>

 
* (참고) 다른 참석자 리뷰 [[링크](https://zzsza.github.io/etc/2019/10/20/tensorflow-kr-3th-meeting/?fbclid=IwAR3bq6T4-LZeEEIjYTbCkhEOwqSBJK1wcY_xywFFJf5LD2muUi7QzWjDhco)]