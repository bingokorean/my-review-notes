# [A survey of named entity recognition and classification](http://nlp.cs.nyu.edu/sekine/papers/li07.pdf) (2007.01)


## [[Review-note]](https://1drv.ms/w/s!AllPqyV9kKUrghOCc3_ZxbGjx1Ao)

## Summary
(survey논문 특징이지만) NER연구들이 시간순으로 어떻게 발전해왔는지 알 수 있었고 (2007년까지), 구체적으로 예시를 들어주면서 설명해주고 있어서 쉽게 이해할 수 있었다. 더불어 NER관점에서 기계학습의 기초(e.g., SL, SSL, USL)도 배울 수 있었다. 특히 NER with SSL의 연구역사 부분이 흥미로웠다.
* NER은 텍스트에서 중요한 entity들(e.g., person, location name, organization, numeric expression, ...)을 찾는 문제로, Information Extraction의 sub-task이다.
* 똑같은 텍스트 데이터이지만 genre(e.g., scientific, informal) 또는 domain(e.g., sports, business)이 바뀌면 성능이 떨어진다고 보고되는 연구들이 있다.
* (카테고리가 많은) NER을 통해 ontology를 만들 수 있다. (물론 상위레벨의 모든 task들도 해당된다. NLP는 계층구조를 가지는 파이프라인 모델이기 때문이다.)
* SSL는 데이터가 부족한 SL을 도와줄 수 있다. 확실한 판단을 가지는 시드 규칙 셋을 이용해서 우리가 분류할 단어를 정의한다. 그리고 모든 데이터속에서 시드 규칙을 만족하는 단어들을 찾고 그들의 context 정보를 수집한다. 수집하면서 확실한 판단을 가지는 context 정보는 문맥 규칙 셋에 정의된다. 문맥 규칙을 통해 시드 규칙을 찾는다. 그리고 이를 반복 (SSL의 (mutual) bootstrapping 기법이다.)
* 즉, 데이터는 적고, 우리가 원하는 데이터의 정말 확실한 clue가 있을 때, SSL을 사용하면 효율적이다. (ex. Person을 찾으려 할때, Mr는 확실한 clue이다.)
* UL은 similarity of context를 통해 clustering/grouping해서 SL에 도움을 준다. 즉, 자주 등장하는 단어들의 쌍, 또는 비슷한 문맥을 가지는 단어들끼리 grouping한다. 그리고 lexical resource인 WordNet을 종종 활용한다.
* Feature vector representation은 abstraction이라 생각할 수 있다.
* 기본적으로 feature design은 해당 데이터의 content와 데이터의 structure를 기반으로 설계된다. (e.g., Document features are defined over both document content and document structure)
* 데이터를 구분짓는 단위가 있으면 적극활용해야 한다. 예를 들어, text는 word-level, sentence-level, document-level, corpus-level 등으로 구분될 수 있다. word-level만 사용해서는 복잡한 문제(e.g., metonymy)를 풀기 어렵다.  
* 인상 깊은 feature는 char패턴함수를 feature로 사용하는 것이었다. (e.g., x="Machine-223.", GetPattern(x)="Aaaaaaa-000-", GetSummarizedPattern(x)="Aa-0-")
* NER 검증방법은 conference(e.g., MUC, IREX, CONLL, ACE)마다 조금씩 다르다. 기본적으로는 micro-averaged f-measure을 사용한다. 단, precision과 recall을 정의하는 방식은 다양하다. 







### Reference
Nadeau, David, and Satoshi Sekine. "A survey of named entity recognition and classification." Lingvisticae Investigationes 30.1 (2007): 3-26.



