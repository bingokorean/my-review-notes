## Overview Text Mining and Analytics

### Text Mining and Analytics

* Text mining과 Text analytics는 서로 비슷한 의미를 가진다. (따로 구분하지는 않는다.) 미묘한 차이는,
   * Mining emphasizes more on the process. 
   * Analytics emphasizes more one the result or having a problem in mind. 
* Text mining과 Text analytics의 의미: (raw text data보다 더 유용한 정보로 만드는 것) 
   * Turning text data into high-quality information (→more concise information about the topic, which might be easier for humans to digest than the raw text data and which minimizes human effort on consuming text data) or actionable knowledge (→emphasize the utility of the information or knowledge we discover from text data and it’s for optimal decision making)
* Text mining/analytics is related to text retrieval, which is an essential component in any text mining system. (text retrieval refers to finding relevant information from a large amount of text data) 
<br>

* Text retrieval은 다음과 같은 두 가지 방향으로 text mining에 좋은 영향을 끼친다. 
   * Text retrieval can be a preprocessor for text mining (big data->most relevant data, which minimizes human efforts)
   * Text retrieval is needed for knowledge provenance(출처). Once we find the patterns or actionable knowledge in text data, we generally would have to verify the knowledge by looking at the original text data. So, the users would have to have some text retrieval support and users can go back to the original text data to interpret the pattern or to better understanding an analogy or to verify whether a pattern is really reliable. 

### Text vs. Non-text Data: Humans as Subjective “Sensors”

* Text data와 non-text data의 유사점(analogy)이 무엇인지 살펴보자. 
* 사람(=subject sensor)을 온도계처럼 physical sensor처럼 생각해보자.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/data-mining-specialization/notes/images/human_sensor.png" width="60%" height="60%"></p>

* (물리적인) 센서들은 세상을 자기들만의 방식으로 보고, 자기들만의 format으로 저장한다. 이들과 유사하게 (주관적인) 센서인 사람들 역시 자기들만의 관점(perspective)에서 세상을 바라보고 text format으로 저장한다. 
* 이렇게 사람을 일종의 센서라고 정의하면 위의 그림과 같이 모든 데이터 종류에 상관없이 하나의 똑같은 framework로 만들 수 있다.

### The General Problem of Data Mining

* Data mining을 해결하는데 위와 같이 (text와 non-text 관계없이) 모든 데이터를 통합적으로 생각하는 방법이 필요하다. 
* Non text data 못지않게 text data도 중요한데, 이유는 text data contain semantic contents, knowledge about the users especially preferences and opinions of users.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/data-mining-specialization/notes/images/data_mining_problem.png" width="60%" height="60%"></p>

* 데이터 마이닝 문제는 모든 데이터를 (문제 해결에 용이한) actionable knowledge (=decision making)로 바꾸는 것이다. 
* Data Mining Software 모듈에는 다양한 데이터 타입을 처리할 수 있는 다양한 종류의 데이터 마이닝 알고리즘들이 존재한다. 

### The Problem of Text Mining

Text mining 문제 역시 일반 데이터 마이닝 문제와 비슷하다. 따라서, text 뿐만 아니라 non-text data도 같이 처리해줘야 한다. (Joint Mining)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/data-mining-specialization/notes/images/non_text_data.png" width="60%" height="60%"></p>

### Landscape of Text Mining and Analytics

* This picture basically covered multiple types of knowledge that we can mine from text data. 
   * 단계가 진행될수록 (1->5) mining하기 어렵다. 
   * 단계들의 결과를 재사용하거나 non-text data를 사용하면 다음 단계들의 knowledge를 잘 mining할 수 있다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/data-mining-specialization/notes/images/text_mining_landscape.png" width="60%" height="60%"></p>

사람마다 자기만의 관점으로 세상을 바라보면서 글로 표현한다. 또한, 같은 사람일지라도 시간에 따라 다른 관점을 가질 수도 있다. So, humans are able to perceive the world from some perspective and then the human sensor would form a view of the world called observed world. Of course, this would be different from the real world because the perspective the person has taken can often be biased. 

Observed world는 entity-relation graph와 같은 knowledge representation language 방식으로 나타낼 수 있다. Then the human would express what the person has observed using a natural language such as English.  

So, the main goal of text mining is actually to revert this process of generating text data. We hope to be able to uncover some aspect in this process. 

   1. Mining knowledge about language: English text data를 보면서 영어의 문법, 패턴 등 찾기
   2. Mining content of text data (mining knowledge about observed world): text data가 (어떤 과점에서 세상을 바라보고) 의미하고 있는 바는 무엇인지, 어떤 (유용한) 정보가 들어있는지 확인
   3. Mining knowledge about the observer: text data를 보면서 observer(글쓴이)의 정보를 파악. 예를 들어, 글쓴이의 심리, 기분(sentiment) 등
      * 주의: observed world와 person을 구분해야 된다. Person은 주관성이 있고, world는 객관성이 있기 때문이다. 하지만, 그래도 person에 의해 생성된 text data에는 약간의 사실(객관성)이 주관성과 같이 섞여 있을 것이다. 
   4. Infer other real-world variables (predictive analytics): we want to predict the value of certain interesting variables. 

빨간색 선의 의미: When we infer other real world variables, we could also use some of the results from mining text data as intermediate results to help the prediction. For example, after we mine the content of text data, we might generate some summary of content and that summary could be used to help us predict the variables of the real world. 어떤 mining의 result를 feature를 사용하든 말든 feature의 근본은 text data이다. 다시 말해, the processing of text data to generate some features that can help with the prediction, is very important!!! -> 다른 mining의 결과가 다른 mining의 새로운 feature가 되어 prediction을 하는데 도움을 줄 수 있다.

Non-text data 역시 prediction task를 처리 하는데 도움을 준다. (당연한 말! 데이터가 다양하고 많을 수록 모델 예측 성능은 높아진다. 단, 유용한 데이터일 경우에만)

Non-text data can be also used for analyzing text by supplying context(문맥). When we look at the text data alone, we will be mostly looking at the context and/or opinions expressed in the text. But, text data generally also has context associated. For example, the time and the location are associated with text data and these are useful context information. 

The context can provide interesting angles for analyzing text data. For example, we might partition text data into different time periods because of the availability of the time. And then we can analyze text data in each time period and then make a comparison. 보통 text data는 context에 속해 있다. 시간과 장소와 같은 non-text data를 같이 활용하면 text data의 여러 측면을 context를 통해 파악할 수 있다.

So, in this sense, non-text data can actually provide interesting angles or perspective for text data analysis. And it can help us make context-sensitive analysis content or the language usage or the opinions about the observer or the authors of text data. We could analyze the sentiment in different contexts. Non-text data를 통해 text data의 다양한 측면을 이해할 수 있다.

### Topics Covered in This Course

위에서 제시한 Mining 방법들이 다음과 같이 연결된다. 여기는 application과 같다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/data-mining-specialization/notes/images/text_mining_app.png" width="60%" height="60%"></p>
