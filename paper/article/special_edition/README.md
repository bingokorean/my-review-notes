# [국립국어원 새국어생활 - [특집] 4차 산업 혁명 시대의 국어 생활](http://www.korean.go.kr/nkview/nklife/2017_4.html)

## Contents

1. [인공 지능 기반 언어 처리 기술 - 자연어 대화 인터페이스를 중심으로](#인공-지능-기반-언어-처리-기술---자연어-대화-인터페이스를-중심으로)
2. 말뭉치와 언어학 (최재웅)
3. 일상생활 속으로 들어온 기계 번역 (김준석)
4. 우리말 자연어 처리 기술: 과거와 현재 (김학수)
5. 음성 언어 처리 기술, 어디까지 왔나 (이경님)

---

## 인공 지능 기반 언어 처리 기술 - 자연어 대화 인터페이스를 중심으로

김경선 (다이퀘스트)


### 1. 들어가는 말

* 인공 지능
   * 미래를 이끌어 갈 가장 큰 키워드
   * 인공지능 기술, 사회 전반에 걸쳐 다양한 분야에 적용
   * 우리가 쉽게 체험할 수 있는 것은 자연어 대화 인터페이스 서비스

* 자연어 대화 인터페이스 서비스
   * 인간과 컴퓨터의 교류 (HCI: Human & Computer Interaction) 기술
      * 사용자의 자연어 발화를 자연어 처리로 이해
      * 언어 이해, 의도 분석, 응답 생성, 대화 처리
   * 빅데이터 분석 (Big Data Analysis) 기술
      * 인간이 생성한 수많은 지식 데이터를 분석하여 사용자의 발화 의도에 적합한 정보 생성
	  * 정보 추출, 지식 추론, 상황 인지 등
	  
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_1.PNG" width="80%" height="80%"></p>

### 2. 인공 지능과 언어 처리 기술

* 인공 지능이란 무엇인가
   * 마쓰오 유타가 <인공 지능과 딥러닝> 책
      * 인공 지능의 역사에 3번의 인공 지능 붐(Boom)과 2번의 겨울이 있었음
	  * 각각의 붐을 '추론 탐색 기술(1차)', '지식(2차)', '기계 학습, 특징 표현 학습(3차)'의 시기로 봄
	  * 인공 지능은 '생각한다'는 것을 실현시키기 위해 추상적인 것을 다루는 학문이라고 정의
   * 여러 전문가들, 다음과 같이 인공 지능을 구분
      * 인지적인 관점에서 '생각하는 시스템'
	  * 공학적인 관점에서 '행동하는 시스템'
      * 오류를 포함하는 인간이냐 합리적인 이성이냐에 따라 구분됨
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_3.PNG" width="80%" height="80%"></p>

* 이 글에서는 '인간처럼 행동하는 시스템'의 관점에서 인공 지능의 다양한 지능 중 언어 지능을 실생활에 적용한 기술인 언어 처리와 자연어 대화 인터페이스 기술을 중심으로 설명

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_4.PNG" width="80%" height="80%"></p>

### 3. 언어 처리 기술: 자연어 처리

* 자연어 처리란 인간의 언어를 기계에서 처리하는 기술
* 자연어 처리의 근간을 이루는 언어 이해 기술은... 
   * '형태소 분석', '구문 분석', '개체명 분석', '화행 분석', '의도 분석'의 과정으로 구성
   * 주로 규칙에 기반 한 방법과 기계 학습을 이용한 방법을 사용 (최근 딥러닝을 많이 사용)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_5.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_6.PNG" width="80%" height="80%"></p>

* 대화 분석은 담화 분석과 비슷하다 (ex. 사용자가 전체적으로 무엇을 말하고 싶어하나?)
* 화행 분석까지는 분석 범위가 현재 문장만을, (ex. 현재 사용자가 질문을 하는 것인가?) 대화 분석(담화 분석)은 현재 문장뿐만 아니라 그 이전의 문장들까지 고려해야 한다. 즉, 문맥을 이해해야 한다.

형태소 분석

* 태뷸러 파싱(Tabular Parsing)을 이용한 '형태소 후보 분석'
   * 기분석 처리, 형태소 사전과 좌우 접속 정보에 기반 한 방법
* 규칙 또는 기계 학습에 기반 한 '품사 부착(Part of Speech Tagging)'
   * 국립국어원에서 배포한 세종 코퍼스를 기본적인 기계 학습 및 추론에 활용
   * HMM(Hidden Markov Model)이나 CRF(Conditional Random Field)와 같은 전통적인 방법을 사용해 왔으나, 최근에는 딥러닝을 적용한 방법들을 사용
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_7.PNG" width="80%" height="80%"></p>

구문 분석

* 문장을 형태소, 명사구 등의 구성 성분으로 분해하여 각 구성 성분 간의 의존 관계와 역할을 분석
* 구 단위로 의존 관계를 분석하는 '구 구조 문법 기반 구문 분석'에서 구성 성분 단위로 분석하는 '의존 문법 기반 구문 분석'으로 발전하고 있다
* 언어별 특성에 따라 한국어는 CKY(Cocke-Younger-Kasami) 알고리즘 방법이, 영어는 MST(Maximum Spanning Tree) 알고리즘 방법이 사용
* 세종코퍼스와 딥러닝을 적용하는 시도가 많아지고 있다

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_8.PNG" width="80%" height="80%"></p>

개체명 인식

* 인명(Person), 지명(Location), 기관명(Organization) 등 의미(개체명 태그)를 문장 구성 성분에 부여하는 것
* 정보 추출, 정보 검색, 질의응답(Question Answer) 등을 위해 사용
* 주로 사회 통념에 따른 일반적인 개체명을 기본으로 하여 각 영역마다 필요한 개체명 계층 구조를 상세하게 정의하여 사용
* 다음 표은 계체명의 예로, 일반적인 조직, 단체 등의 개체명 정의 외에 은행에서 발급받은 공인 인증서 사용 범위 설명 등을 위한 특별한 목적에 따라 금융 기관, 은행, 증권사 등을 나누어 사용하고 있다

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_table1.PNG" width="80%" height="80%"></p>

* 개체명 인식을 위해서 전통적으로 순서적 인식(Sequence Labeling)에서 좋은 성능을 발휘하는 CRF(Conditional Random Field) 알고리즘을 많이 사용하나,
* 최근에는 딥러닝을 적용하여 개선한 LSTM(Long Short Term Memory)-CRF 방법이나 LSTM-RNN(Recurrent Neural Network)을 널리 사용한다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_9.PNG" width="80%" height="80%"></p>

화행(Speech Ant)(의도(Intention)) 분석
   
* 자연어 처리에서 화행 분석은 주로 발화에 나타난 범용적인 의도를 나타내며, 영역 지식이 더해진 구체적인 의도를 의도 분석이라고 구분하는 것이 일반적이다. 즉, 화행 분석은 general하고 의도 분석은 specific하다.
* 화행(의도) 분석은 개체명 분석이나 정보 분류 방법과 유사한 프로세스로 구성된다.
* 화행(의도) 분석은 주로 대화 처리를 위해 사용되기 때문에 대화가 사용될 영역이나 응답 형태에 따라 각각에 적합한 화행과 의도 정보 체계를 정의해야 한다 (Domain Specific!)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_10.PNG" width="80%" height="80%"></p>

감성 분석

* 감성 분석은 문장에 나타난 긍정과 부정 표현을 분석하는 것으로,
* 사용자의 의견을 나타내므로 빅데이터 분석 분야의 오피니언 마이닝에서 많이 사용
* 주로 SNS, 블로그 등에 나타난 사용자의 의견을 분석하여 마케팅이나 개발 기능 도출 등에 많이 사용
* 사용자의 언어 사용 패턴에 의해 결정되므로 규칙 기반이 주로 사용 (최근에는 딥러닝)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_11.PNG" width="80%" height="80%"></p>

* 감성 분석은 화행 분석을 넘어 담화 분석의 범위에 포함되지 않을까 생각한다. 사용자의 감성은 현재 문장뿐만 아니라 말하고자 하는 전체 내용을 토대로 분석하는 것이 필요하다. (현재 문장은 부정적이나 전체적으로 긍정을 표현할 수도 있다)
   
정보 추출

* 정보 추출은 텍스트 형태의 비정형 데이터로부터 정형화된 정보를 추출하는 기술로, 
* 추출된 정형 정보는 데이터베이스에 저장하거나 온톨로지로 변환한 후 시맨틱 저장소에 저장하여 빅데이터 분석이나 시맨틱 검색 등의 용도로 사용
* 정형 정보로 변환하기 위해서는 언어 사용 패턴을 활용한 '규칙 기반 방법'이나, LDA(Latent Dirichlet Allocation) 등의 '기계 학습 기반 방법'이 주로 사용

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_1213.PNG" width="80%" height="80%"></p>

### 4. 자연어 대화 인터페이스: 자연어 처리 기술의 적용

* 최근 자연어 처리 기술이 가장 많이 적용되는 분야는 구글의 어시스턴트, 아마존의 에코, 삼성전자의 빅스비 등으로 대표되는 자연어 대화 인터페이스 분야임
* 자연어 대화 인터페이스는 초기에는 개인의 디바이스나 서비스 기능 활용, 관심사 검색, 채팅 등을 지원하는 개인 비서봇이 주류를 이뤘으나, 현재는 금융이나 쇼핑 등의 도메인에서 상품에 대한 설명이나 구매, 예약 등을 지원하는 상담봇의 형태로 상용화되고 있다
* 또한, 주 대상도 스마트폰 서비스 위주에서 로봇, 스마트 스피커, 자동차에 이르기까지 다양한 분야로 확산

#### 4.1. 자연어 대화 인터페이스를 위한 기술

* 자연어 대화 인터페이스에서 처리하는 대화 유형
   * '채팅형 대화'
      * 사회적 관습이나 지식, 대화 모델을 바탕
	  * 유행어를 사용하여 재미있게 응답하는 농담식으로 대화하는 것
   * '목적 지향형 대화'
      * 사용자의 발화 의도를 이해하고 발화의 목적에 따라 적합한 발화 또는 행위
	  * 사용자의 발화 의도를 사용자 발화 텍스트뿐만 아니라 착용 가능한(웨어러블) 기기의 체온, 위치 등의 센서 데이터와 같은 주변 정보를 이용하여 발화의 맥락을 이해하고, 부유하고 있는 다양한 지식 베이스를 이용해 응답을 생성하는 것 
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_15.PNG" width="80%" height="80%"></p>

* 자연어 대화 인터페이스 
   1. '언어 이해': 텍스트 형태의 발화를 형태소, 개체명, 화행(의도) 분석기 등의 언어 분석기를 이용하여 사용자 의도를 파악
   2. '대화 관리': 적합한 대화 모델을 이용하여 응답을 위한 의미 구조를 생성
   3. '응답 생성': 추론과 수행 엔진을 이용하여 지식 베이스로부터 적합한 응답을 생성
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_16.PNG" width="80%" height="80%"></p>
   
언어 이해

* 언어 이해는 사용자 발화를 언어적으로 분석하여 사용자의 의도를 기계가 이해할 수 있는 의미 구조로 변환하는 과정. 
* 주로 자연어 처리 기본 기술을 사용.

대화 관리

* 대화 관리는 사용자 발화의 의미 구조를 기반으로 적합한 대화 모델을 이용해 시스템 응답의 의미 구조를 생성하는 과정.
* 주로 목적 기반 대화 모델(Goal Oriented Dialog Model)이 많이 사용.
* 목적을 잘 수행하기 위해서는 어떤 대화 모델을 사용할 것인가가 관건.
* 대화 모델은 사용되는 추론 방법에 따라 '규칙 기반 대화 모델'과 '통계 기반 대화 모델'로 구분.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_17.PNG" width="80%" height="80%"></p>

* 계획(Plan) 기반 대화 모델
   * 가장 전통적인 모델로, 이후에 나온 다른 대화 모델들의 기반이 됨.
   * 전체적인 목적 수행을 대화에서 사용되는 용도를 기준으로 영역 계획(Domain Plan), 문제 해결 계획(Problem-Solving Plan), 담화 계획(Discourse Plan)을 나누어 정의하는 것으로써 모델의 유연성을 확보.
   * 이전 스크립트(Script) 기반 대화 모델의 단점인 도메인 이식성 문제도 극복.
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_18.PNG" width="80%" height="80%"></p>

* 액티브 온톨로지(Active Ontology) 기반 대화 모델
   * 과업(Task)을 액티브 온톨로지 기반의 개념(Concept)들로 나누어 정의하고, 언어 이해 결과와 추론 기반으로 대화 서비스를 진행하며, 수행 엔진 모듈을 이용해 외부 데이터 연계나 서비스를 수행.
   * 애플사의 시리(Siri) 개발에 사용됨.
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_19.PNG" width="80%" height="80%"></p>

* 예제 기반 대화 모델(Example Based Dialog Model)
   * 최근에 많이 연구되고 있는 통계 기반 방법론.
   * 실용적이며 개발 편이성이 높음.
   * 대량의 대화 예제(Dialog Example)로부터 언어 분석을 이용해 대화 예제 색인(Index)을 구성하고 이로부터 사용자 발화와 가장 유사한 예제를 찾아 이를 시스템 응답에 이용함.
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_20.PNG" width="80%" height="80%"></p>

* 강화 학습을 사용하는 POMDP 기반 대화 모델
   * 통계 기반 대화 모델 중 하나
   * 강화 학습(Reinforcement Learning)을 사용하는 POMDP(Partially Observable Markov Decision Process) 기반 대화 모델.
   * 대화 학습 말뭉치를 마코프 결정 프로세스(MDP)를 통해 강화 학습 한 대화 모델을 이용하여 최적의 응답을 생성하는 방법.
   * 최근에는 알파고에서 사용된 딥러닝 기반 강화 학습인 DQN(Deep Q-Network)이 개방되면서 이를 이용한 연구가 많이 진행중.
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_21.PNG" width="80%" height="80%"></p>

응답 생성

* 응답 생성은 시스템 응답 의미 구조로부터 사용자에게 전달될 응답 발화, 행위 등을 생성하는 것.
* 응답 발화 생성을 위해서는 발화 템플릿을 만들어 놓고 적절한 정보를 결합시키는 템플릿 기반 방법이 일반적으로 사용됨.
* 지식 베이스를 이용한 정답 생성을 위해 IBM사의 왓슨(Watson) 등에서는 자식 추론 엔진을 사용.
* 다양한 외부 데이터와 서비스의 연계를 위해 애플사의 시리 등에서는 서비스 수행 엔진을 사용.
   
* 지식 추론 엔진
   * 다양한 지식 원천으로부터 정답 추론에 사용한 지식 베이스를 구축하고, 추론을 통해 사용자에게 적절한 응답과 확률을 생성함.
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_22.PNG" width="80%" height="80%"></p>

* 서비스 수행 엔진
   * 사용자의 발화에 적합한 응답과 서비스를 수행하기 위해 각 서비스별로 수행에 필요한 외부 데이터, 서비스 URL, 속성 등의 정보를 정의하여 이용함.
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_23.PNG" width="80%" height="80%"></p>

#### 4.2. 자연어 대화 인터페이스

* 스마트 디바이스 + 개인 비서 서비스(=자연어 대화 인터페이스)
  * 애플 + 시리
  * 삼성전자 + 에스보이스(S-voice)
  * 엘지전자 + 큐보이스(Q-voice)
  * 구글 + 구글나우(Google Now)
  * 마이크로소프트 + 코타나(Cortana)
  * 페이스북 + M
  * 초기에 얼마나 사람 친화적이고, 얼마나 사람이 만족할 만한 것인가에 초점이 맞추어졌으나, 점차 지식 정보를 활용하는 부분으로 관심이 이동.
  * 주로 자체 인프라를 이용해 시장에 진입하면서 국내 제조업체보다는 글로벌 서비스 기업을 중심으로 경쟁이 심화되고 있음.
   
<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/paper/article/special_edition/images/1_25.PNG" width="80%" height="80%"></p>

* 사물 인터넷(IoT) + 자연어 대화 인터페이스
  * 아마존(에코) + 알렉사(Alexa)
  * 에코는 스피커를 기반으로 스마트 홈 음성 서비스를 제공하는 저가 보급형 인공 지능 디바이스로, 스마트 홈을 위한 사물 인터넷 허브를 지향함.
  * 에코는 알렉사라는 음성 자연어 대화 인터페이스를 이용하여 연계된 사물 인터넷 서비스에 대한 조절과 음악 재생, 피자 주문, 렌터카 요청 등 다양한 외부 서비스 활용을 지원함.
   
* 로봇 + 자연어 대화 인터페이스
  * 지보(Jibo)
     * 보급형으로 만들어진 가정용 인공 지능 로봇으로 메시지 전송, 사진 촬영, 영상 전화 등 스마트 홈 구축에 필수적인 기능들을 갖춤.
	 * 스마트 홈을 위한 사물 인터넷 허브 역할을 지향.
	 * 사용자의 표정을 읽어 감정을 인식하는 기능, 자연어 대화 인터페이스를 이용한 사용자 친화적인 인터페이스 등을 제공.
  * 소프트뱅크 - 패퍼(Pepper)
     * 가정 및 업소용 인공 지능 로봇
	 * 자연어 대화 인터페이스로 IBM의 왓슨을 사용
	 * 최근에는 얼굴 인식을 통해 고객의 인상, 연령, 성별 등을 파악
	 * 마이크로소프트사의 빅데이터 분석 시스템인 애저(Azure)를 활용하여 고객에게 맞는 최적의 상품을 찾아주기도 함.
	 * 소매점 서비스, 접수와 관광 안내, 노인 돌봄 및 의료 서비스, 교육 서비스 등으로 영역을 확장.
   
* 문자 + 자연어 대화 인터페이스
  * 카카오뱅크 - 24시간 로봇 상담 서비스
  * 은행, 증권사, 카드사에 이르는 많은 금융 기업들이 자연어 대화 인터페이스를 이용한 자동 상담을 도입하고 있음.
  * 콜센터를 운영하는 다른 기업과 기관드롤 확장.
  * 국외에서는 넥스트 아이티사(Next IT Inc.)를 중심으로 항공 예약, 콜센터 상담, 금융 상담 등 다양한 영역에 문자를 활용한 지능형 자동 상담이 서비스되고 있음.
   
### 5. 맺는말

* 인공 지능, 빅데이터 기술과 이를 실생활에서 사용할 수 있도록 적용한 자연어 처리, 자연어 대화 인터페이스를 알아봄.
* 애플사의 시리를 시작으로 재조명받기 시작한 자연어 대화 인터페이스는 최근 스마트폰, 태블릿, 자동차, 로봇 등에 이르는 다양한 기기로 확대되어 인간과 기계를 연계하는 핵심 기술로 사용되고 있음.
* 그러나, 자연어 대화 인터페이스 기술은 아직 초기 단계로 그 성능이 사용자의 기대 수준보다 낮은 경우가 많음.
* 이를 해결하기 위해서, 기술적 개선도 필요하지만, 기존에 보유하고 있는 데이터 외의 영역에서 다양한 현상을 포함하는 언어 데이터 구축이 필요. 그리고 이를 딥러닝과 같은 최신 기계 학습에서 사용할 수 있도록 변환하는 일이 필수.
* 이러한 언어 데이터 구축 및 변환은 컴퓨터 공학자뿐만 아니라 언어 전문가와 영역별 전문가들이 협업을 해야만 완성도 있는 언어 데이터 구축이 가능. 
* 그런데 개인이나 개별적인 회사가 독자적으로 추진하기는 어려움. 따라서, 국가 기관을 중심으로 전문가들이 협업하여 국문 모두가 사용할 수 있는 언어 데이터를 만들고 공개해야 함.
* 또한, 국민 모두가 검증하고 개선해 나가는 절차가 도입되어야 함.

