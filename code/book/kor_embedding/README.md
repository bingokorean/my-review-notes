# 한국어 임베딩

[책](https://ratsgo.github.io/natural%20language%20processing/2019/09/12/embedding/) <br>
[깃허브](https://github.com/ratsgo/embedding) <br>
이기창


### Contents

<div id='top.'/>

1. [서론](#1.)
2. [벡터가 어떻게 의미를 가지게 되는가](#2.)
3. [한국어 전처리](#3.)
4. 


<br>

<div id='1.'/>

## 1. 서론

### 1.1. 임베딩이란

- 컴퓨터는 인간이 사용하는 자연어(natural language)를 그대로 이해하는 것이 아닌 숫자를 계산한다.
    - 기계의 자연어 이해와 생성은 연산(computation)이나 처리(processing)의 영역이다.
- 표현력이 무한한 언어를 컴퓨터가 연산할 수 있는 숫자로 바꿀 수 있을까?
    - 말과 글을 숫자로 변환할 때 어떤 정보를 함축시킬 것인가? 정보 압축 과정에서 손실은 발생하지 않는가? 그 손실은 어떻게 줄일 것인가?
- 자연어 처리(Natural Language Processing) 분야에서 임베딩(Embedding)이란?
    - 사람이 쓰는 자연어를 기계가 이해할 수 있는 숫자의 나열인 벡터(vector)로 바꾼 결과 혹은 그 일련의 과정 전체를 의미한다.
    - 단어나 문장 각각을 벡터로 변환해 벡터 공간(vector space)으로 ‘끼워 넣는다(embed)’는 의미이다.
- 가장 간단한 형태의 임베딩은 단어의 빈도를 그대로 벡터로 사용하는 것이다.
    - 단어-문서 행렬(Term-Document Matrix)

### 1.2. 임베딩의 역할

#### 1.2.1. 단어/문장 간 관련도 계산

- 단어를 벡터로 임베딩하는 순간 단어 벡터들 사이의 유사도(similarity)를 계산하는 일이 가능해진다.

#### 1.2.2. 의미/문법 정보 함축

- 임베딩은 벡터인 만큼 사칙연산이 가능하다.
- 단어 벡터 간 덧셈/뺄셈을 통해 단어들 사이의 의미적, 문법적 관계를 도출해낼 수 있다.
- ‘아들’ - ‘딸' + ‘소녀’ = ‘소년' 이 성립하면 성공적인 임베딩이라 볼 수 있다.
- ‘아들’ - ‘딸’ 사이의 관계와 ‘소년' - ‘소녀’ 사이의 의미 차이가 임베딩에 함축돼 있으면 품질이 좋은 임베딩이라 볼 수 있다.
- 이렇게 단어 임베딩을 평가하는 방법을 단어 유추 평가(Word Analogy Text)라 한다.

#### 1.2.3. 전이 학습

- 품질 좋은 임베딩은 NLP 태스크를 위한 모델의 정확도와 학습 속도를 높일 수 있다.
    - ex. Random 임베딩보다 FastText 임베딩을 사용하는 모델의 성능이 더 높다.
    - ex. Random 임베딩보다 FastText 임베딩을 사용하는 모델의 학습(수렴) 시간이 더 빠르다.
- 임베딩을 다른 딥러닝 모델의 입력값으로 쓰는 기법을 전이 학습(transfer learning)이라고 한다.

### 1.3. 임베딩 기법의 역사와 종류

#### 1.3.1. 통계 기반에서 뉴럴 네트워크 기반으로

- 초기 임베딩 기법은 대부분 말뭉치의 통계량을 직접적으로 활용하는 경향이 있었다.
- 대표적인 기법이 잠재 의미 분석(Latent Semantic Analysis)이다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_1_1.png" width="60%" height="60%"></p>

- 단어-문서 행렬 특징
    - 행의 개수가 말뭉치 전체의 어휘(vocabulary) 수와 같으므로 매우 많다.
    - 행렬 대부분의 요소 값은 0이므로 희소 행렬(sparse matrix) 특징을 가진다.
- 단어-문서 행렬에 잠재 의미 분석(LSA)을 실시한다.
    - 단어(1)를 기준으로, 그리고 문서(2)를 기준으로 행렬을 분해할 수 있다.
    - (1)은 단어 임베딩, (2)는 문서 임베딩이 된다.
- 잠재 의미 분석 수행 대상 행렬은 여러 종류가 될 수 있다.
    - 단어-문서 행렬
    - TF-IDF 행렬 - Term Frequency-Inverse Document Frequency
    - 단어-문맥 행렬 - Word-Context Matrix
    - 점별 상호 정보량 행렬 - Pointwise Mutual Information Matrix
<br>

- Neural Network 기반의 임베딩 기법 다음과 같은 문제를 학습하면서 생성된다.
    - 이전 단어들이 주어졌을 때 다음 단어가 뭐가 될지 예측
    - 문장 내 일부분에 구멍을 뚫어 놓고(masking) 해당 단어가 뭐가 될지 예측

#### 1.3.2. 단어 수준에서 문장 수준으로

- 2017년 이전의 임베딩 기법들은 대게 단어 수준 모델이었다. (ex. NPLM, Word2Vec, GloVe, FastText, Swivel 등)
    - 단어 임베딩 기법들은 각각의 벡터에 해당 단어의 문맥적 의미를 함축한다.
    - 단어 임베딩 기법의 단점은 동음이의어(homonym)를 분간하기 어렵다는 점이다.
        - 단어의 형태가 같다면 동일한 단어로 보고, 모든 문맥 정보를 해당 단어 벡터에 투영한다.
<br>

- 2018년 초 ELMo(Embeddings from Language Models)가 발표된 이후 문장 수준 임베딩 기법이 주목을 받았다.
    - BERT(Bidirectional Encoder Representations from Transformer), GPT(Generative Pre-Training) 등이 있다.
    - 문장 수준 임베딩 기법은 개별 단어가 아닌 단어 시퀀스(sequence) 전체의 문맥적 의미를 함축하기 때문에 단어 임베딩 기법보다 전이 학습 효과가 좋다.
    - 예를 들어, 먹는 ‘배’(pear), 신체 부위인 ‘배’(belly), 교통 수단의 '배’(ship) 등 다양한 의미를 지닌 동음이의어가 있다고 하자. 
        - 단어 임베딩은 이 모든 의미가 뭉뚱그려져 하나로 표현되지만, 문장 임베딩을 사용하면 이들을 분리해 이해할 수 있다.

#### 1.3.3. 룰 → 엔드투엔드 → 프리트레인/파인 튜닝

- 1990년대까지의 자연어 처리 모델은 대부분 사람이 피처(feature)를 직접 뽑았다.
    - 피처 추출은 언어학적인 지식(규칙)을 활용한다.
        - 한국어에서는 명사 앞에 관형사가 온다. 명사 뒤에 조사가 온다. 동사(어간) 앞에 부사가 오고, 뒤에는 어미가 온다.
- 2000년대 중반 이후 자연어 처리 분야에서도 딥러닝 모델이 주목을 받기 시작했다.
    - 딥러닝 모델은 입력(input)과 출력(output) 사이의 관계를 잘 근사(approximate)하므로 사람이 모델에 규칙을 굳이 직접 알려주지 않아도 된다.
    - 데이터를 통째로 모델에 넣고 입출력 사이의 관계를 사람의 개입 없이 모델 스스로 처음부터 끝까지 이해하하도록 유도한다. → 이러한 기법을 엔드투엔드 모델(end-to-end model)이라 한다.
- 2018년 ELMo 모델이 제안된 이후 엔드투엔드에서 프리트레인(pre-train)과 파인 튜닝(fine tuning)으로 발전하고 있다.
<br>

- 다운스트림 태스크(downstream task)
    - 우리가 풀고 싶은 자연어 처리의 구체적인 문제
        - 품사 판별(Part-Of-Speech tagging), 개체명 인식(Named Entity Recognition), 의미역 분석(Semantic Role Labeling) 등..
- 업스트림 태스크(upstream task)
    - 다운스트림 태스크에 앞서 해결해야 할 과제라는 뜻이다.
        - 단어/문장 임베딩을 pre-train하는 작업
<br>

- 다운스트림 태스크 예시
    - 품사 판별
        - 나는 네가 지난 여름에 한 [일]을 알고 있다. → 일: 명사(noun)
    - 문장 성분 분석
        - 나는 [네가 지난 여름에 한 일]을 알고 있다. → 네가 지난 여름에 한 일: 명사구(noun phrase)
    - 의존 관계 분석
        - [자연어 처리는] 늘 그렇듯이 [재미있다]. → 자연어 처리는, 재미있다: 주격명사구(nsubj)
    - 의미역 분석
        - 나는 [네가 지난 여름에 한 일]을 알고 있다. → 네가 지난 여름에 한 일: 피행위주역(patient)
    - 상호 참조 해결
        - 나는 어제 [성빈이]를 만났다. [그]는 스웨터를 입고 있었다. → 그=성빈이

#### 1.3.4. 임베딩의 종류와 성능


- 임베딩 기법 3가지
    - 행렬 분해 방법
    - 예측 방법
    - 토픽 기반 방법
<br>

- 행렬 분해 기반 방법
    - 행렬 분해(factorization) 방법 - GloVe, Swivel
        - 원래 행렬을 두 개 이상의 작은 행렬로 쪼개는 방식
        - 분해한 이후엔 둘 중 하나의 행렬만 쓰거나 둘을 더하거나(sum), 이어 붙여(concatenate) 임베딩으로 사용한다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_1_2.png" width="60%" height="60%"></p>

- 예측 기반 방법
    - 주로 뉴럴 네트워크 방법 - Word2Vec, FastText, BERT, ELMo, GPT
    - 다음과 같은 과정에서 학습하는 방법이다.
        - 어떤 단어 주변에 특정 단어가 나타날지 예측하거나, 
        - 이전 단어들이 주어졌을 때 다음 단어가 무엇엇일지 예측하거나,
        - 문장 내 일부 단어를 지우고 해당 단어가 무엇일지 맞추거나,
<br>

- 토픽 기반 방법
    - 잠재 티리클레 할당(Latent Dirichlet Allocation)이 대표적인 기법
        - 주어진 문서에 잠재된 주제(latent topic)를 추론(inference)하는 방식으로 임베딩을 학습한다.
        - LDA 모델은 학습이 완료되면 각 문서가 어떤 주제 분포(topic distribution)을 갖는지 확률 벡터 형태로 반환한다.
<br>

- 임베딩 성능 평가
    - Tenney et al. (2019) - 영어 기반 다운스트림 태스크에 대한 임베딩 종류별 성능을 분석
    - 형태소 분석, 문장 성분 분석, 의존 관계 분석, 의미역 분석, 상호 참조 해결 등
    - 파인 튜닝 모델의 구조를 고정한 뒤 각각각의 임베딩을 전이 학습시키는 형태로 정확도를 측정
    - 문장 임베딩 기법(ELMo, GPT, BERT)가 단어 임베딩 기법(GloVe)를 크게 앞선다.

### 1.4. 이 책이 다루는 데이터와 주요 용어

- 말뭉치(corpus)
    - 임베딩 학습이라는 특정한 목적을 가지고 수집한 표본(sample)
    - 자연어의 표현력은 무한하다. 말뭉치이더라도 표본임을 인지하자.
- 컬렉션(collection)
    - 말뭉치에 속한 각각의 집합
    - 한국어 위키백과와 네이버 영화 리뷰를 말뭉치로 쓴다면 이들 각각이 컬렉션이다.
- 문장(sentence)
    - 이 책에서 다루는 데이터의 기본 단위
    - 문장은 생각이나 감정을 말과 글로 표현할 때 완결된 내용을 나타내는 최소의 독립적인 형식 단위
    - 그러나 실제로는 이렇게 엄밀하게 분리할 수는 없다. 처리해야되는 데이터가 많으므로.
    - 단순히 마침표(.), 느낌표(!), 물음표(?)와 같은 기호로 구분된 문자열을 문장으로 취급
- 문서(document)
    - 생각이나 감정, 정보를 공유하는 문장 집합
    - 문서는 단락(paragraph)의 집합으로 표현될 수도 있으나 여기에서는 문서와 단락을 굳이 구분하지 않는다.
    - 단순히 줄바꿈(‘\n’) 문자로 구분된 문자열을 문서로 취급한다.
- 토큰(token)
    - 가장 작은 단위
    - 문맥에 따라서 토큰을 단어(word), 형태소(morpheme), 서브워드(subword)라고 부를 수 있지만, 같은 뜻으로 서술하는 것이니 별도의 언급이 없다면 이들 용어 의미 차이에 크게 신경 쓰지 않아도 된다.
- 어휘 집합(vocabulary)
    - 말뭉치에 있는 모든 문서를 문장으로 나누고 여기에 토크나이징을 실시한 후 중복을 제거한 토큰들의 집합
    - 어휘 집합에 없는 토큰을 미등록 단어(unknown word)라고 한다.

### 1.5. 이 장의 요약

* 임베딩이란 자연어를 기계가 이해할 수 있는 숫자의 나열인 벡터로 바꾼 결과 혹은 그 일련의 과정 전체를 가리킨다.
* 임베딩을 사용하면 단어/문장 간 관련도를 계산할 수 있다.
* 임베딩에는 의미적/문법적 정보가 함축돼 있다.
* 임베딩은 다른 딥러닝 모델의 입력값으로 쓰일 수 있다.
* 임베딩 기법은 (1) 통계 기반에서 뉴럴 네트워크 기반으로 (2) 단어 수준에서 문장 수준으로 (3) 엔드투엔드에서 프리트레인/파인 튜닝 방식으로 발전해왔다.
* 임베딩 기법은 크게 행렬 분해 모델, 예측 기반 방법, 토픽 기반 기법 등으로 나눠진다.
* 이 책에서 다루는 데이터의 최소 단위는 토큰이다. 문장은 토큰의 집합, 문서는 문장의 집합, 말뭉치는 문서의 집합을 가리킨다. 토크나이즈란 문장을 토큰으로 분석하는 과정을 의미한다. 어휘 집합은 말뭉치에 있는 모든 문서를 문장으로 나누고 여기에 토크나이즈를 실시한 후 중복을 제거한 토큰들의 집합이다.

[[top](#top.)]

<br>

<div id='2.'/>

## 2. 벡터가 어떻게 의미를 가지게 되는가

### 2.1. 자연어 계산과 이해

- 임베딩에 자연어 의미를 어떻게 함축할 수 있을까?
- 비결은 자연어의 통계적 패턴(statistical pattern) 정보를 통째로 임베딩에 넣는 것이다.
- 자연어의 의미는 해당 언어 화자들이 실제 사용하는 일상 언어에서 드러나기 때문이다.
<br>

- 임베딩을 만들 때 쓰는 통계 정보는 크게 세 가지가 있다.
    - 첫째: 문장에 어떤 단어가 (많이) 쓰였는지, (백오브워즈 가정)
        - 저자의 의도는 단어 사용 여부나 그 빈도에서 드러난다고 가정한다. 여기서 단어의 순서(order) 정보는 무시한다.
    - 둘째: 단어가 어떤 순서로 등장하는지, (언어 모델)
        - 언어 모델은 단어의 등장 순서를 학습해 주어진 단어 시퀀스가 얼마나 자연스러운지 확률을 부여한다. (백오브워워즈 가정과 대비된다)
    - 셋째: 문장에 어떤 단어가 같이 나타났는지, (분포 가정 - distributional hypothesis)
        - 단어의 의미는 그 주변 문맥(context)을 통해 유추해볼 수 있다고 보는 것이다.

| 구분 | 백오브워즈 가정 | 언어 모델 | 분포 가정 |
|:---|:---:|:---:|:---:|
| 내용 | 어떤 단어가 (많이) 쓰였는가 | 단어가 어떤 순서로 쓰였는가 | 어떤 단어가 같이 쓰였는가 |
| 대표 통계량 | TF-IDF | - | PMI (Pointwise Mutual Information) |
| 대표 모델 | Deep Averaging Network | ELMo, GPT  | Word2Vec |

- 위의 세 철학은 서로 연관이 있다.
- 언어 모델에서는 단어의 등장 순서를, 분포 가정에서는 이웃 단어(문맥)를 우선시한다.
- 어떤 단어가 문장에서 주로 나타나는 순서는 해당 단어의 주변 문맥과 떼려야 뗼 수 없는 관계를 가진다.
- 한편 분포 가정에서는 어떤 단어 쌍(pair)이 얼마나 자주 나타나는지와 관련한 정보를 수치화하기 위해 개별 단어 그리고 단어 쌍의 빈도 정보를 적극 활용한다.
- 요컨대 백오브워즈 가정, 언어 모델, 분포 가정은 말뭉치의 통계적 패턴을 서로 다른 각도에서 분석하는 것이며 **상호 보완적**이다.

### 2.2. 어떤 단어가 많이 쓰였는가

#### 2.2.1. 백오브워즈 가정

- 수학에서 백(bag)이란 중복 원소를 허용한 집합(multiset)을 뜻한다. 원소의 순서는 고려하지 않는다.
- 자연어 처리 분야에서 백오브워즈(bag of words)란 단어의 등장 순서에 관계없이 문서 내 단어의 등장 빈도를 임베딩으로 쓰는 기법을 말한다.
- 백오브워즈 임베딩에는 ‘저자가 생각한 주제가 문서에서의 단어 사용에 녹아 있다’는 가정이 깔려 있다. 다시 말해 주제가 비슷한 무서라면 단어 빈도 또는 단어 등장 여부 역시 비슷할 것이고, 백오브워즈 임베딩 역시 유사할 것이라고 보는 것이다.
- 백오브워즈 임베딩은 간단한 아이디어이만 정보 검색(information retrieval) 분야에서 여전히 많이 쓰이고 있다.

#### 2.2.2. TF-IDF

- TF-IDF 역시 단어 등장 순서를 고려하지 않는다는 점에서 백오브워즈 임베딩이라고 이해할 수 있다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_1.png" width="70%" height="70%"></p>

- TF(word): 특정 document에서 해당 word의 Term Frequency 
- N: 전체 말뭉치에서 전체 document 개수
- DF(word): 전체 말뭉치에서 해당 word가 포함된 document 개수
- DF가 클수록 다수 문서에 쓰이는 범용적인 단어이다.
- TF는 같은 단어라도 문서마다 다른 값을 가진다.
- DF는 문서가 달라지더라도 단어가 같다면 동일한 값을 가진다.
- IDF가 클수록 특이한 단어라는 뜻이다. 이는 단어의 주제 예측 능력(해당 단어만 보고 문서의 주제를 가늠해볼 수 있는 정도)과 직결된다.

#### 2.2.3. Depp Averaging Network

* Deep Averaging Network(Iyyer el al., 2015)는 백오브워즈 가정의 뉴럴 네트워크 버전이다. (TF-IDF도 백오브워즈 가정에 속한다는 것을 잊지 말자)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_1_3.png" width="80%" height="80%"></p>

- Iyyer et al. (2015)가 백오브워즈 가정과 연결 지점은 단어의 순서를 고려하지 않는다는 점이다.
- 최종 문장 임베딩은 모든 단어의 임베딩의 평균을 취한다.
- 벡터의 덧셈은 교환 법칙이 성립하기 때문에 평균을 취하는 것은 순서를 고려하지 않는 것과 같다.
- lyyer et al. (2015)은 문장 내에 어떤 단어가 쓰였는지, 쓰였다면 얼마나 많이 쓰였는지 그 빈도만을 따진다.
- 간단한 구조의 아키텍쳐임에도 성능이 좋아서 현업에서도 자주 쓰인다.

### 2.3. 단어가 어떤 순서로 쓰였는가

#### 2.3.1. 통계 기반 언어 모델

- 언어 모델(language model)이란 단어 시퀀스에 확률(probability)을 부여(assign)하는 모델이다.
- 단어의 등장 순서를 무시하는 백오브워즈와 달리 언어 모델은 시퀀스 정보를 명시적으로 학습한다.
- 백오브워즈의 대척점에 언어 모델이 있다고 볼 수 있다.
- 문법적으로나 의미적으로 결함이 없는 훌륭한 한국어 문장이메도 말뭉치에 없는 문장이라면? 언어 모델에서는 해당 문장을 확률을 0을 부여하여 말이 되지 않는 문장으로 취급한다.
<br>

* ‘Chance favors the prepared’라는 표현 다음에 ‘mind’라는 단어가 나타날 확률을 조건부확률(Conditional Probability)의 정의를 활용해 최대우도추정법(Maximum Likelihood Estimation)으로 유도하면 다음 식과 같다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_2.png" width="80%" height="80%"></p>

- 이렇게 전체 시퀀스를 대상으로 확률을 구하면 시퀀스가 길어질수록 말뭉치에서 해당 시퀀스가 없는 경우가 많다.
- 분자, 분모에 있는 시퀀스 Count가 1개로라도 없으면 확률은 0이 되어 무의미한 값이 되어 버린다.
<br>

- n-gram 모델을 사용하면 이런 문제를 일부 해결할 수 있다.
- 직전 n-1개 단어의 등장 확률로 전체 단어 시퀀스 등장 확률을 근사(approximation)하는 것이다.
- 이는 한 상태(state)의 확률은 그 직전 상태에만 의존한다는 마코프 가정(Markov Assumption)에 기반한 것이다.
- ‘Chance favors the prepared mind’라는 단어 시퀀스가 나타날 확률을 바이그램 모델로 근사하면 다음과 같다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_3.png" width="80%" height="80%"></p>

- 바이그램 모델을 일반화한 식은 다음과 같다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_4.png" width="65%" height="65%"></p>

- 바이그램 모델을 통해 많은 시퀀스의 경우를 생성할 수 있다.
- 하지만, 바이그램이라 하더라도 말뭉치에 한 번도 등장하지 않을 수도 있다.
<br>

- 이를 위해 백오프(back-off), 스무딩(smoothing) 등의 방식이 제안됐다.
- 백오프란 n-gram 등장 빈도를 n보다 작은 범위의 단어 시퀀스 빈도로 근사하는 방식이다. n을 크게 하면 할수록 등장하지 않는 케이스가 많아질 가능성이 높다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_5.png" width="55%" height="55%"></p>


- alpha와 beta는 실제 빈도와의 차이를 보정해주는 파라미터이다.
<br>

- 스무딩은 등장 빈도 표에 모두 k만큼을 더하는 기법이다. 이 때문에 Add-k 스무딩 이라고 부르기도 한다.
- 만약 k를 1로 설정한다면 이를 특별히 라플라스 스무딩(Laplace Smoothing)이라고 한다.
- 스무딩을 시행하면 높은 빈도를 가진 문자열 등장 확률을 일부 깍고 학습 데이터에 전혀 등장하지 않는 케이스들에는 아주 작으나마 일부 확률을 부여한다.

#### 2.3.2. 뉴럴 네트워크 기반 언어 모델

* 통계 기반 언어 모델은 단순히 단어들의 빈도를 세어서 학습한다.
* 뉴럴 네트워크는 입력과 출력 사이의 관계를 학습하면서 언어 모델을 구축한다.
* 뉴럴 네트워크는 주어진 단어 시퀀스를 가지고 다음 단어를 맞추는(prediction) 과정에서 학습된다.
* 학습이 완료되면 이들 모델의 중간 혹은 말단 계산 결과물을 단어나 문장의 임베딩으로 활용한다. (ex. ELMo, GPT 등)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_2_1.png" width="55%" height="55%"></p>

* 마스크 언어 모델(masked language model)은 언어 모델 기반 기법과 큰 틀에서 유사하지만 조금 다르다.
* 다음 그림처럼 문장 중간에 마스크를 씌워 놓고, 해당 마스크 위치에 어떤 단어가 올지 예측하는 과정에서 학습한다.
* 언어 모델 기반 기법은 단어를 순차적으로 입력받아 다음 단어를 맞춰야 하기 때문에 태생적으로 일방향(uni-directional)이다.
* 마스크 언어 모델 기반 기법은 문장 전체를 다 보고 중간에 있는 단어를 예측하기 때문에 양방향(bi-directional) 학습이 가능하다.
* 이 때문에 마스크 언어 모델 기반의 방법들은 기존 언어 모델 기법들 대비 임베딩 품질이 더 좋다. BERT가 이 부류에 속한다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_2_2.png" width="45%" height="45%"></p>


### 2.4. 어떤 단어가 같이 쓰였는가

#### 2.4.1. 분포 가정

* 자연어 처리에서 분포(distribution)란 특정 범위, 윈도우(window) 내에 동시에 등장하는 이웃 단어 또는 문맥(context)의 집합을 가리킨다.
* 개별 단어의 분포는 그 단어가 문장 내에서 주로 어느 위치에 나타나는지, 이웃한 위치에 어떤 단어가 자주 나타나는지에 따라 달라진다.
* 어떤 단어 쌍(pair)이 비슷한 문맥 환경에서 자주 등장한다면 그 의미(meaning) 또한 유사할 것이라는 게 분포 가정(distributional hypothesis)의 전제이다.
* 분포 가정은 "단어의 의미는 곧 그 언어에서의 활용이다(the meaning of a word is its use in the language)"라는 언어학자 비트겐슈타인(1889-1951)의 철학에 기반해 있다.
* 즉, 모국어 화자들이 해당 단어를 실제 어떻게 사용하고 있는지 문맥(주변 단어)을 살핌으로써 그 단어의 의미를 밝힐 수 있다는 이야기이다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_2_3.png" width="65%" height="65%"></p>

* '빨래', '세탁'이라는 단어의 의미를 전혀 모른다고 하자.
* 두 단어의 의미를 파악하기 위해서는 이들 단어가 실제 어떻게 쓰이고 있는지 관찰하면 된다.
* 분포 가정을 적용해 유추해보면 '빨래'와 '세탁'은 서로 비슷한 이웃 단어('청소', '요리', '물', '옷/속옷')을 가지므로 비슷한 의미를 가질 가능성이 있다.
* 아울러 타깃 단어인 '빨래'가 이웃 단어들('청소', '요리', '물', '속옷')과 직간접적으로 관계를 지닐 가능성도 있다.
* 그러나, 개별 단어의 분포 정보와 그 의미 사이에는 논리적으로 직접적인 연관성이 있어 보이지는 않는다. 즉, 분포 정보가 곧 의미라는 분포 가정에 의문점이 있다.
* 다음 절에서 분포 정보와 의미 사이에 어떤 관계가 있는지 언어학적 관점에서 살펴보자.

#### 2.4.2. 분포와 의미 (1): 형태소

* 언어학에서 형태소(morpheme)란 의미를 가지는 최소 단위를 말한다. 더 쪼개면 뜻을 잃어버린다.
* 이때의 '의미'는 어휘적인 것뿐만 아니라 문법적인 것도 포함된다.
* '철수가 밥을 먹었다'에서 한국어 화자가 직관적으로 유추해낼 수 있는 형태소 후보는 '철수', '밥' 등이다.
   * '철수'를 '철'과 '수'로 쪼개면 '철수'라는 사람을 지칭하는 의미가 사라진다.
   * '밥'을 'ㅂ'과 '압'으로 나누면 먹는 밥이라는 뜻이 없어진다.
   * 이런 점에서 '철수'와 '밥'은 형태소라 할 수 있다.
<br>

* 언어학자들이 형태소를 분석하는 방법은 조금 다르다. 대표적인 기준으로 계열 관계(paradigmatic relation)이 있다.
   * 계열 관계는 해당 형태소 자리에 다른 형태소가 '대치'돼 쓰일 수 있는가를 따지는 것이다.
   * 언어학자들이 한국어 말뭉치를 다량 분석한 결과 '철수' 자리에 '영희' 같은 말이 올 수 있고, '밥' 대신 '빵'을 쓸 수도 있다는 사실을 확인했다고 가정하자.
   * 언어학자들은 이를 근거로 '철수'와 '밥'에 형태소 자격을 부여한다.
<br>

* 언어학자들이 계열 관계를 바탕으로 형태소를 분석한다는 사실을 곱씹어보자.
* 이는 언어학자들이 특정 타깃 단어 주변의 문맥 정보를 바탕으로 형태소를 확인한다는 이야기와 일맥상통한다.
* 말뭉치의 분포 정보와 형태소가 밀접한 관계를 이루고 있다는 것이다.

#### 2.4.3. 분포와 의미 (2): 품사

* 품사란 단어를 문법적 성질의 공통성에 따라 언어학자들이 몇 갈래로 묶어 놓은 것이다.
* 학교 문법에 따르면 품사 분류 기준은 기능(function), 의미(meaning), 형식(form) 등 세 가지이다.
* 다음 예문을 보자.
   * 이 샘의 깊이가 얼마냐?
   * 저 산의 높이가 얼마냐?
   * 이 샘이 깊다.
   * 저 산이 높다.
<br>

* 기능은 한 단어가 문장 가운데서 다른 단어와 맺는 관계를 말한다.
   * '깊이'와 '높이'는 문장의 주어로 쓰이고 있고, '깊다', '높다'는 서술어로 사용되고 있다.
   * 이처럼 기능이 같은 단어 부류를 같은 품사로 묶을 수 있다.
* 의미란 단어의 형식적 의미를 나타낸다. (i.e. 어떤 단어가 사물의 이름, 움직임, 성질, 상태를 나타내는가?)
   * 어휘적 의미를 기준으로 묶으면 '깊이'와 깊다'를 하나로 묶고, '높이'와 '높다'를 같은 군집으로 넣을 수 있다.
   * 품사 분류에는 어휘적 의미보다는 형식적 의미가 중요하다.
   * 형식적 의미를 기준으로 묶으면 '깊이'와 '높이'를 하나로, '깊다'와 '높다'를 한 덩어리로 묶을 수 있다.
 * 형식이라고 함은 단어의 형태적 특징을 의미한다.
   * '깊이', '높이'는 변화하지 않는다.
   * '깊다', '높다'는 '깊었다'/'높았다', '깊겠다'/'높겠다' 따위와 같이 어미가 붙어 여러 가지 모습으로 변할 수 있다.
   * 이 기준으로 봐도 '깊이', '높이'를 한 덩어리로, '깊다', '높다'를 다른 덩어리로 묶을 수 있다.
<br>

* 실제 품사를 분류할 때는 여러 가지 어려움이 있다.
* 예컨대 의미는 품사 분류 시 고려 대상이 될 수 있으나 결정적인 분류 기준이 될 수 없다.
* 다음 예문을 보자.
   * `공부하다`
   * `공부`
* 한국어 화자라면 대개 '공부하다'를 동사, '공부'를 명사로 분류할 것이다.
   * 즉, '공부하다'는 움직임을 나타내고, '공부'는 사물의 이름이라는 의미를 내포한다고 보는 것이다.
   * 그렇다면 '공부'라는 단어에는 움직임이라는 의미가 전혀 없는 것일까? 딱 잘라 그렇다고 말하기 어렵다.
   * 의미가 품사 분류의 결정적인 기준이 될 수 없다는 이야기다.
* 품사 분류 시 결정적 기준이 될 수 없는 건 형태도 마찬가지다.
* 다음 예문을 보자.
   * `영수가 학교에 간다.`
   * `영수! 조용히 해.`
* 첫 번째에서 '영수'는 명사, 두 번째에서 '영수'는 감탄사로 쓰였다. 형태는 같지만 기능과 의미가 달라졌음을 확인할 수 있다.
<br>

* 그렇다면 품사 분류에서 가장 중요한 기준은 무엇일까? 언어학자들이 꼽는 결정적인 기준은 바로 '기능'이라고 한다.
* 해당 단어가 문장 내에서 점하는 역할에 초점을 맞춰 품사를 분류한다는 것이다.
* 그런데 한국어를 비롯한 많은 언어에서는 어떤 단어의 기능이 그 단어의 분포(distribution)와 매우 밀접한 관련을 맺고 있다고 한다.
* 국어학의 창시자 격인 최현배 선생은 1930년 '조선어 품사분류론'이라는 책에서 다음과 같이 언급한 바 있다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_2_4.png" width="60%" height="60%"></p>

* 기능은 특정 단어가 문장 내에서 어떤 역할을 하는지, 분포는 그 단어가 어느 자리에 있는지를 나타낸다.
* 비유컨대 '이웃사촌'은 정이 들어 사촌 형제나 다를 바 없이 지내는 이웃을 뜻한다.
   * 자주 만나고 가까이에 있는 이웃(분포)이 혈육 같이 챙겨주는 역할(기능)을 하는 데서 생겨난 말이다.
* 이처럼 기능과 분포는 개념적으로 엄밀히 다르지만, 둘 사이에는 밀접한 관련을 지닌다.
* 국어학자들은 한국어 품사 분류의 일반적인 기준을 다음과 같이 정의하고 있다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_2_5.png" width="60%" height="60%"></p>

* 요컨대 형태소의 경계를 정하거나 품사를 나누는 것과 같은 다양한 언어학적 문제는 말뭉치의 분포 정보와 깊은 관계를 갖고 있다.
* 이 덕분에 임베딩에 분포 정보를 함축하게 되면 해당 벡터에 해당 단어의 의미를 자연스레 내재시킬 수 있게 된다.

#### 2.4.4. 점별 상호 정보량

* 점별 상호 정보량(PMI; Poinwise Mutual Information)은 두 확률변수(random variable) 사이의 상관성을 계량화하는 단위이다.
* 두 확률변수가 완전히 독립(independent)인 경우 그 값이 0이 된다.
* 독립이라고 함은 단어 A가 나타나는 것이 단어 B의 등장할 확률에 전혀 영향을 주지 않고, 단어 B 등장이 단어 A에 영향을 주지 않는 경우를 말한다.
* 반대로 단어 A가 등장할 때 단어 B와 자주 같이 나타난다면 PMI 값은 커진다.
* 요컨대 PMI는 두 단어의 등장이 독립일 때 대비해 얼마나 자주 같이 등장하는지를 수치화한 것이다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_6.png" width="55%" height="55%"></p>

* PMI는 분포 가정에 따른 단어 가중치 할당 기법이다. 두 단어가 얼마나 자주 같이 등장하는지에 관한 정보를 수치화했다.
* 이렇게 구축한 PMI 행렬의 행 벡터 자체를 해당 단어의 임베딩으로 사용할 수도 있다.
* 다음 그림은 단어-문맥 행렬(word-context matrix)을 구축하는 과정을 개념적으로 나타낸 것이다.
* PMI 행렬은 다음 그림의 단어-문맥 행렬에 PMI 수식을 적용한 결과이다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_2_7.png" width="60%" height="60%"></p>

* 위의 그림에서 볼 수 있듯 윈도우(window)가 2라면 타깃 단어 앞뒤로 2개의 문맥 단어의 빈도를 계산한다.
* 모든 단어를 훑어 단어-문맥 행렬을 모두 구했다고 해보자. 전체 빈도 수는 1000회, '빨래'가 등장한 횟수는 20회, '속옷'이 등장한 횟수는 15회, '빨래'와 '속옷'이 동시에 등장한 빈도는 10회라 해보자.
* '빨래'-'속옷'의 PMI 값은 다음과 같다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_7.png" width="55%" height="55%"></p>


#### 2.4.5. Word2Vec

* 분포 가정의 대표적인 모델은 2013년 구글 연구 팀이 발표한 Word2Vec이라는 임베딩 기법이다.
* CBOW 모델은 문맥 단어들을 가지고 타깃 단어 하나를 맞추는 과정에서 학습된다.
* Skip-gram 모델은 타깃 단어를 가지고 문맥 단어가 무엇일지 예측하는 과정에서 학습된다.
* 둘 모두 특정 타깃 단어 주변의 문맥, 즉 분포 정보를 임베딩에 함축한다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_2_6.png" width="60%" height="60%"></p>

* 실제 Word2Vec 기법은 PMI 행렬과 깊은 연관이 있다는 논문이 발표되기도 했다.

### 2.5. 이 장의 요약

* 임베딩에 자연어의 통계적 패턴(statistical pattern) 정보를 주면 자연어의 의미(semantic)를 함축할 수 있다.
* 백오브워즈 가정에서는 어떤 단어의 등장 여부 혹은 그 빈도 정보를 중시한다.
* 백오브워즈 가정의 대척점에는 언어 모델이 있다. 언어 모델은 단어의 등장 순서를 학습해 주어진 단어 시퀀스가 얼마나 자연스러운지 확률을 부여한다.
* 분포 가정에서는 문장에서 어떤 단어가 같이 쓰였는지를 중요하게 따진다.
* 백오브워즈 가정, 언어 모델, 분포 가정은 말뭉치의 통계적 패턴을 서로 다른 각도에서 분석하는 것이며 상호 보완적이다.

[[top](#top.)]

<br>

<div id='3.'/>

## 3. 한국어 전처리

### 3.1. 데이터 확보

#### 3.1.1. 한국어 위키백과

```
$ bash dumpdata.sh
$ python parse_wiki.py
$ python tokenize_wiki.py
```

```shell
## dumpdata.sh

echo "download ko-wikipedia..."
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -P /notebooks/embedding/data/raw
mkdir -p /notebooks/embedding/data/processed
```
```python
## parse_wiki.py
# xml 데이터의 text 태그 데이터를 파싱.

from gensim.corpora import WikiCorpus
from gensim.utils import to_unicode

"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
"""

in_f = "notebooks/embedding/data/raw/kowiki/latest/kowiki-latest-pages-articles.xml.bz2"
out_f = "notebooks/embedding/data/processed/processed_wiki_ko.txt"

"""Convert Wikipedia xml dump file to text corpus"""
output = open(out_f, 'w', encoding = "utf-8")
wiki = WikiCorpus(in_f, tokenizer_func=tokenize, dictionary=Dictionary())
i = 0
for text in wiki.get_texts():
    output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
    i = i + 1
    if (i % 10000 == 0):
        print('Processed ' + str(i) + ' articles')
output.close()
print('Processing complete!')
```
```python
## wiki_tokenizer.py
# 불필요한 문자열 (ex. 특수문자, 목차, 주소) 제거

import re
from gensim.utils import to_unicode

WIKI_REMOVE_CHARS = re.compile("'+|(=+.{2,30}=+)|__TOC__|(ファイル:).+|:(en|de|it|fr|es|kr|zh|no|fi):|\n", re.UNICODE)
WIKI_SPACE_CHARS = re.compile("(\\s|゙|゚|　)+", re.UNICODE)
EMAIL_PATTERN = re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.UNICODE)
WIKI_REMOVE_TOKEN_CHARS = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)

def tokenize(content, token_min_len=2, token_max_len=100, lower=True):
    content = re.sub(EMAIL_PATTERN, ' ', content)  # remove email pattern
    content = re.sub(URL_PATTERN, ' ', content) # remove url pattern
    content = re.sub(WIKI_REMOVE_CHARS, ' ', content)  # remove unnecessary chars
    content = re.sub(WIKI_SPACE_CHARS, ' ', content)
    content = re.sub(MULTIPLE_SPACES, ' ', content)
    tokens = content.replace(", )", "").split(" ")
    result = []
    for token in tokens:
        if not token.startswith('_'):
            token_candidate = to_unicode(re.sub(WIKI_REMOVE_TOKEN_CHARS, '', token))
        else:
            token_candidate = ""
        if len(token_candidate) > 0:
            result.append(token_candidate)
    return result
```

* 참고 - [wikiextractor](https://github.com/attardi/wikiextractor) - 위키백과 정제 라이브러리

#### 3.1.2. KorQuAD

* [KorQuAD](https://korquad.github.io)는 한국어 기계 독해(Machine Reading Comprehension)를 위한 데이터셋
* LG CNS가 구축. 2018년 공개. 7만 79건.
* 한국어 위키백과의 '알찬 글', '좋은 글' 등 양질의 문서를 수집해 이 가운데 일부 문단으로부터 파생될 수 있는 질문과 답변 쌍을 만듦.
* KorQuAD는 구축 전 과정에 사람들이 직접 개입. 검증 역시 철저히 함. 이 때문에 한국어 임베딩용 말뭉치로 손색이 없음.
```
$ bash dumpdata.sh
$ python preprocess_korquad.py
```
```shell
## dumpdata.sh

echo "download KorQuAD data..."
wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -P /notebooks/embedding/data/raw
wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -P /notebooks/embedding/data/raw
mkdir -p /notebooks/embedding/data/processed
```
```python
## preprocess_korquad.py
corpus_fname = "/notebooks/embedding/data/raw/KorQuAD_v1.0_train.json"
output_fname = "notebooks/embedding/data/processed_korquad_train.txt"

with open(corpus_fname) as f1, open(output_fname, 'w', encoding='utf-8') as f2:
    dataset_json = json.load(f1)
    dataset = dataset_json['data']
    for article in dataset:
        w_lines = []
        for paragraph in article['paragraphs']:
            w_lines.append(paragraph['context'])
            for qa in paragraph['qas']:
                q_text = qa['question']
                for a in qa['answers']:
                    a_text = a['text']
                    w_lines.append(q_text + " " + a_text)
        for line in w_lines:
            f2.writelines(line + "\n")
```

#### 3.1.3. 네이버 영화 리뷰 말뭉치

* 네이버 영화 페이지의 영화 리뷰들을 평점과 함꼐 수록한 한국어 말뭉치
* 박은정님께서 구축. 감성 분석, 문서 분류 태스크 수행에 사용.
* 레코드 하나는 문서(리뷰)에 대응.
* 문서 ID, 문서 내용, 레이블로 구성. 각 열은 탭 문자로 구분.
* 데이터 20만개. 절반은 긍정. 나머지 절반은 부정.

```
$ bash dumpdata.sh
$ python preprocess_korquad.py
```
```shell
## dumpdata.sh

echo "download naver movie corpus..."
wget https://github.com/e9t/nsmc/raw/master/ratings.txt -P /notebooks/embedding/data/raw
wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt -P /notebooks/embedding/data/raw
wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt -P /notebooks/embedding/data/raw
mkdir -p /notebooks/embedding/data/processed
```
```python
corpus_path = "/notebooks/embedding/data/raw/ratings.txt"
output_fname = "/notebooks/embedding/data/processed/processed_ratings.txt"

def process_nsmc(corpus_path, output_fname, process_json=True, with_label=True):
    if process_json:
        file_paths = glob.glob(corpus_path + "/*")
        with open(output_fname, 'w', encoding='utf-8') as f:
            for path in file_paths:
                contents = json.load(open(path))
                for content in contents:
                    sentence = content['review'].strip()
                    if len(sentence) > 0:
                        f.writelines(sentence + "\u241E" + content['movie_id'] + "\n")
    else:
        with open(corpus_path, 'r', encoding='utf-8') as f1, \
                open(output_fname, 'w', encoding='utf-8') as f2:
            next(f1)  # skip head line
            for line in f1:
                _, sentence, label = line.strip().split('\t')
                if not sentence: continue
                if with_label:
                    f2.writelines(sentence + "\u241E" + label + "\n")
                else:
                    f2.writelines(sentence + "\n")
```

### 3.2. 지도 학습 기반 형태소 분석

* 한국어는 조사와 어미가 발달한 교착어(agglutinative language)이기 때문에 문장이나 단어의 경계 처리를 섬세하게 해야 한다.
* '가겠다', '가더라', '가겠더라', '가다' 를 그대로 어휘 집합에 넣기 보다는 '가', '겠', '다', '더라' 로 형태소 단위로 구성하는 게 좋다. 
* 형태소로 어휘 집합을 구성하면 다양한 활용어들을 최소한의 핵심적인 어휘들로 인지할 수 있다.
* 교착어인 한국어는 한정된 종류의 조사와 어미를 자주 이용하기 때문에 각각에 대응하는 명사, 용언(형용사, 동사), 어간만 어휘 집합에 추가하면 취급 단어 개수를 꽤 줄일 수 있다.
* 즉, 형태소 분석을 잘해야 자연어 처리의 효율성을 높일 수 있다.

#### 3.2.1. KoNLPy 사용법

* [KoNLPy](http://konlpy.org/en/latest)는 은전한닢, 꼬꼬마, 한나눔, Okt, 코로나 등 5개 오픈소스 형태소 분석기를 파이썬 환경에 제공한다.

```python
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    elif tokenizer_name == "khaiii":
        tokenizer = KhaiiiApi()
    else:
        tokenizer = Mecab()
    return tokenizer

tokenizer = get_tokenizer("komoran")
tokenizer.morphs("아버지가방에들어가신다")
tokenizer.pos("아버지가방에들어가신다")
```

#### 3.2.2. Khaii 사용법





















