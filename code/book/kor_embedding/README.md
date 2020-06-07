# 한국어 임베딩

[책](https://ratsgo.github.io/natural%20language%20processing/2019/09/12/embedding/) <br>
[깃허브](https://github.com/ratsgo/embedding) <br>
이기창


### Contents

<div id='contents'/>

1. [서론](#1.)
2. [벡터가 어떻게 의미를 가지게 되는가](#2.)
3. 

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

<p align="center"><img src="http://www.sciweavers.org/upload/Tex2Img_1591496037/eqn.png" width="70%" height="70%"></p>

- TF(word): 특정 document에서 해당 word의 Term Frequency 
- N: 전체 말뭉치에서 전체 document 개수
- DF(word): 전체 말뭉치에서 해당 word가 포함된 document 개수
- DF가 클수록 다수 문서에 쓰이는 범용적인 단어이다.
- TF는 같은 단어라도 문서마다 다른 값을 가진다.
- DF는 문서가 달라지더라도 단어가 같다면 동일한 값을 가진다.
- IDF가 클수록 특이한 단어라는 뜻이다. 이는 단어의 주제 예측 능력(해당 단어만 보고 문서의 주제를 가늠해볼 수 있는 정도)과 직결된다.

#### 2.2.3. Depp Averaging Network

* Deep Averaging Network(Iyyer el al., 2015)는 백오브워즈 가정의 뉴럴 네트워크 버전이다. (TF-IDF도 백오브워즈 가정에 속한다는 것을 잊지 말자)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/pic_1_3.png" width="60%" height="60%"></p>

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

<p align="center"><img src="http://www.sciweavers.org/upload/Tex2Img_1591496675/eqn.png" width="70%" height="70%"></p>

- 이렇게 전체 시퀀스를 대상으로 확률을 구하면 시퀀스가 길어질수록 말뭉치에서 해당 시퀀스가 없는 경우가 많다.
- 분자, 분모에 있는 시퀀스 Count가 1개로라도 없으면 확률은 0이 되어 무의미한 값이 되어 버린다.
<br>

- n-gram 모델을 사용하면 이런 문제를 일부 해결할 수 있다.
- 직전 n-1개 단어의 등장 확률로 전체 단어 시퀀스 등장 확률을 근사(approximation)하는 것이다.
- 이는 한 상태(state)의 확률은 그 직전 상태에만 의존한다는 마코프 가정(Markov Assumption)에 기반한 것이다.
- ‘Chance favors the prepared mind’라는 단어 시퀀스가 나타날 확률을 바이그램 모델로 근사하면 다음과 같다.

ddd

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_1_1.png" width="60%" height="60%"></p>

dddd

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2.png" width="100%" height="100%"></p>

ddd

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_2.png" width="60%" height="60%"></p>

ddd

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/kor_embedding/images/math_2_3.png" width="80%" height="80%"></p>
