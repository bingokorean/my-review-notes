## Natural Language Content Analysis

Natural Language Content Analysis is the foundation of text mining. In particular, natural language processing with a factor with how we can represent text data. And this determines what algorithms can be used to analyze and mine text data. 

### Basic Concepts in NLP

한 문장을 예를 들어서, NLP의 기본 개념들을 살펴보자.
다음과 같은 문장을 보았을 때, (영어권) 사람은 특별한 고민 없이 잘 이해할 수 있다. 반면, 컴퓨터는 다음과 같은 문장을 이해하기 위해서 몇 단계의 프로세스를 거쳐야 한다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/data-mining-specialization/notes/images/nlp_process.PNG" width="60%" height="60%"></p>

1. Tokenization: knowing what are the words: how to segment the words in English / 그냥 공백 기준으로 나누기만 하면 된다.
2. Lexical analysis (Part-of-speech tagging): knowing the syntactical categories of the token words / for example dog is a noun, chasing is a verb, etc / 단어 단위의 프로세싱
3. Syntactic analysis (Parsing): figuring out the relationship between the token words. / for example, a dog is a noun phrase, one the playground would be a prepositional phrase, etc / 의미를 표현하기 위해 서로 연결하는 방법은 다양하다. (단어보단 구가 더 의미적이다.) / the outcome is a parse tree that tells us the structure of the sentence / 구 단위의 프로세싱

But this is not semantics yet. So, in order to get the meaning, we have to map these phrases and these structures into some real world entities that we have in our mind. For example, dog is a concept that we know. So, connecting these phrases with what we know is understanding. 
Text data로부터 우리가 만들어낸 structure와 우리가 현실세계에서 알고 있는 지식들을 mapping해야 의미적인 분석(semantic analysis)을 할 수 있다.
 
4. Semantic analysis: 구조와 우리의 지식을 연결시키는 과정 / Now, a computer would have to formally represent these entities by using symbols. For example, Dog(d1), Boy(b1), Chasing(d1, b1, p1) 이렇게 token word들을 각각 함수로 나타내면서 semantic을 표현할 수 있다. 함수의 인자 개수에 따라 의미정보 크기가 달라진다 / 주어+동사+목적어 단위의 프로세싱
5. Inference: 각각의 함수들을 사용해 기존에 우리가 정해놓은 규칙에 의해 if, then형식으로 추론할 수 있다. / For example, if we assume there’s a rule that if someone is being chased, then the person can get scared, then we can infer this boy might be scared. / if then 규칙이 필요하기 때문에 additional knowledge를 사용할 수 밖에 없다.
6. Pragmatic analysis (speech act): further infer what this sentence is requesting or why the person is saying the sentence / understanding the purpose of saying the sentence. / in this sentence, a person saying this may be reminding another person to bring back the dog. / Pragmatic analysis를 하는 이유는 어떤 사람이 글을 쓸 때, 저자만의 의도가 항상 있다고 가정하기 때문이다. 

### NLP is Difficult!

#### 사람 입장

The main reason why NLP is very difficult is because it’s designed to make human communication efficient. 사람들간에 의사소통을 할 때, 언어의 경제성 때문에 생략되는 함축적인 의미들이 많다. As a result,
   * We tend to omit a lot of common sense knowledge because we assume all of us have this knowledge and there is no need to encode this knowledge (it makes communication efficient)
   * We tend to keep a lot of ambiguities (ambiguities of words because we assume we have the ability to disambiguate the word. So, there’s no problem with having the same word to mean possibly different things in different context.  

#### 컴퓨터 입장

Yet for a computer these would be very difficult because a computer does not have the common sense knowledge that we do. So, the computer will be confused indeed. And this makes it hard for NLP. 실질적으로 앞에서 보았던 단계들 모두 컴퓨터에게는 어려운 task이다. 단계들은 물론이고 특히 다음과 같은 두 가지 문제 때문에 더 어렵다.
   * Ambiguity is a main killer! Meaning that in every step, there are multiple choices and the computer would have to decide the right choice or decision. 
   * Common sense reasoning is pre-required in order to fully understand NL. And the computers today don’t yet have that. 

### Examples of Challenges

* Word-level ambiguity
   * “design” can be a noun or a verb (ambiguous POS)
   * “apple” has multiple meanings (ambiguous sense)
* Syntactic ambiguity
   * “natural language processing” (modification)
   * 구조적인(syntactic) 측면에서 두 가지 의미로 해석될 수 있다. 
   * processing of natural language 와 language processing is natural, 두 가지 혹은 여러 가지 중에서 어떤 것을 선택해야 할까?
   * “A man saw a boy with a telescope” (Propositional Phrase Attachment)
   * 전치사구가 주어에 달라붙는지 목적어에 달라붙는지 애매하다. 전치사를 어디에 붙여야(attach) 할까?
<br>

* Anaphora resolution: “John persuaded Bill to buy a TV for himself.” (himself=John or Bill?)
* Presupposition: “He has quit smoking” implies that he smoked before.

### The State of the Art NLP 알고리즘 성능 

위와 같은 문제점 때문에 (특히 ambiguity) the-state-of-the-art 알고리즘이라 하더라도 정확하게 처리하지 못한다. (2015년 기준) 간단한 task인 POS도 100% 정확도를 못 가진다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/data-mining-specialization/notes/images/nlp_score.PNG" width="60%" height="60%"></p>

심지어 특정 데이터셋에 한정해서 성능평가를 했기 때문에 위의 수치들이 잘 나온 것이라고 볼 수 있다. 데이터셋을 일반화시키면 위와 같은 성능은 나오지 않을 것이다. 

Parsing은 부분적으로는 잘 되지만 하나의 문장을 통째로 잘 Parsing하는 것은 아직까지 큰 어려움을 가지고 있다. Semantic analysis는 다양한 종류가 있는데, 그 중 Entity/relation extraction은 NER과 같다. 어떤 단어가 사람인지, 장소인지 파악하는 것이다. Inference는 매우 어려운 task이다. 이를 big domain에서는 할 수 없고, 매우 제한된 domain에서만 실행할 수 있다. 이처럼 일반화시키는 것이 일반적인 인공지능의 문제이다. Speech act analysis 역시 매우 어려운 문제라서 사람의 도움, 질 좋은 데이터 도움을 받아 특별한 경우에만 실행할 수 있다. 

#### Statistical method의 필요성

컴퓨터는 자연어를 명확히 이해하는 것은 거의 불가능이다. 따라서, 텍스트 마이닝이 어렵고, 우리는 mechanical한 방법론이나 computational method에 의지할 순 없다. 

we have to use statistical machine learning method of statistical analysis methods to try to get as much meaning out from the text as possible. Later, you will see that there are actually many statistical algorithms that can indeed extract interesting knowledge from text even though we can’t fully understand meaning of all the natural language sentences precisely. 

### What we can’t do

* 100% POS tagging
   * “He turned off the highway.” vs “He turned off the fan.” 여기서 off라는 전치사의 의미가 context마다 syntactic category가 다르게 사용될 것이다.
* General complete parsing
   * “A man saw a boy with a telescope.” Telescope를 가진 사람이 man인지 boy인지 애매하다.
* Precise deep semantic analysis
   * Will we ever be able to precisely define the meaning of “own” in “John owns a restaurant”?

The state of art in NLP can be summarized as follows. 

> Robust and general NLP tends to be shallow while deep understanding doesn't scale up

For this reason, the techniques that we cover in this course are shallow techniques for analyzing or mining text data and they’re generally based on statistical analysis (shallow analysis). (전통적인) 통계적 방법론들의 한계점은 shallow analysis를 잘 할 수 있지만, deep analysis는 다소 힘들다는 점이다. 

Shallow techniques have the advantage of being able to applied to any text data in any natural language about any topic. But, the downside is that they don’t give us a deeper understanding of text. For that, we have to rely on deeper natural language analysis. 

Deep understanding을 위해서는 human effort를 많이 사용하여 많은 양의 라벨 데이터를 확보해야 한다. 이들을 사용하여 기계학습 모델을 통한 감독 학습을 실시하면 deep understanding을 실현할 수 있다.

In practical applications, we generally combine the two kinds of techniques with the general statistical and methods as a backbone as the basis. These can be applied to any text data. And on top of that, we’re going to use humans to annotate more data and to use supervised learning to do some tasks as well as we can especially for those important tasks to bring humans into the loop to analyze text data more precisely. 

But, this course will cover the general statistical approaches that don’t require much human effort. So, they’re practically more useful than deeper analysis techniques that require a lot of human effort to annotate text data. 

* Shallow technique
   * Based on statistical analysis
   * Unsupervised learning
   * No need human efforts
   * Can do Robust and general NLP
* Deeper understanding technique
   * Supervised learning
   * Need human efforts to annotate data
   * Can’t do Robust and general NLP
   * But can do deep understanding


### Summary

* NLP is the foundation for text mining. Obviously the better we can understand the text data, the better we can do text mining. 
* Computers are far from being able to understand natural language
   * Deep NLP requires common sense knowledge and inferences, thus only working for very limited domains, not feasible for large scale text mining.
   * Shallow NLP based on statistical methods can be done in large scale and is thus more broadly applicable to a lot of applications. 
* In practice: statistical NLP as the basis, while humans provide help as needed in various ways.
 










