# Developing Korean chatbot 101
Tensorflow Korea Conference 2 <br>
조재민


## [Review-note](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!277&ithint=file%2cdocx&app=Word&authkey=!AM9UyjW1BQSsZUk)

## Summary
chat-bot의 전체적인 윤곽을 그려주고 있고 chat-bot을 개발하기 위한 것들을 잘 설명해주고 있다.
* 단어(언어)라는 것을 (오랫동안) update된 representation이라고 표현하고 있다. 그리고 언어는 사람의 추상적인 concept/thought를 추상적으로 표현해주는 approximator 함수라고 표현하고 있다. 다시 말해, text data는 image data처럼 pure하지 않고 information loss가 있는 data라고 말하고 있다.
* 내 생각은 언어라는 것은 경제성/효율성에 의해 끊임없이 재생성되기 때문에 단어라는 text data 역시 함축적인 의미를 많이 가지고 있다. 그러한 함축적인 의미를 뽑아낼려면 context information을 활용하는게 필수적이다.
* 인간의 언어를 완벽하게 이해할려면 인간 수준의 knowledge base를 구축해야 된다. 현재는 최소한으로 의사소통할 수 있는 Ontology 정도밖에 갖추지 못하고 있다. 그리고, 딥러닝 방법(e.g., end-to-end) 역시 아직 역부족이다.
* 잘 동작하는 chat-bot을 만들기 위해서 task의 범위를 한정짓는게 필요하다. 그리고 시스템을 여러 단계(modular)로 나눠서 각개격파를 해야한다. (물론 질좋고 task에 맞는 데이터양이 제일 중요하다.)
* feature design은 word embedding와 여러가지 문제(e.g., WSD)를 해결하기 위한 feature들을 서로 concatenation 시켰다. WSD를 해결하기 위해 POS embedding을 사용. word와 함께 POS를 같이 embedding 시켰다는 말인데, pos처럼 애초에 다른 정보들도 미리 index나 tagging해놓고 word와 같이 embedding시켜버릴 수는 없을까?
* word2vec의 한계점: 이웃단어들이 비슷한 단어들끼리 비슷하게 매핑할 수 있는데, 비슷한 단어들끼리 (e.g., football vs. baseball) (의미상으로) 구분하진 못한다. 여기서는 단어 index를 따로 넣어서 구분짓고 있다.
* NLP는 하위 단계의 NLP(e.g., POS, NER)의 성능이 매우 중요하다. 왜냐하면 상위 단계의 NLP(e.g, sentimant analysis, machine translation)는 하위 단계의 NLP 결과를 feature로 사용하기 때문이다.
* 한국어 NLP는 Awesome-Korean-NLP github을 참조하자.

