# 2.1 Composition and Recursion in Symbol Systems

## Keyword


## Summary
### 1. Composition
The essense of cognitivism is represented well by the principle of **compositionality** (i.e., the meaning of the whole is a function of the meaning of the parts)

The compositionality is expounded by Gareth Evans in regard to **language**. He asserts that `the meaning of a complex expression is determined by the meanings of its constituent expressions and the rules used to combine them`. That is, sentences are composed from sequences of words.

Compositionality를 반대로 생각해보자. `The whole can be decomposed into reusable parts or primitives`. It is applicable to other faculties such as action generation. In motor schemata theory (Nucgaek Arbib (1981)), complex, goal-directed actions can be decomposed into sequences of behavior primitives (=sets of commonly used behavior pattern segments).

Cognitive scientists have found a good analogy between:
  * the compositionality of mental processes like combining the meanings of words into those of sentences or combining the images of behavior primitives into those of goal-directed actions "at the back of our mind"
  * the computaional mechanics of the combinatorial operations of operands

In both cases, we have concrete objects (**symbols**) and **distinct procedures** for manipulating them in our brains. Since these objects to be manipulated either by computers or in mental processes, are **symbols** without any physical dimensions such as weight, length, speed, or force, their manipulation processes are considered to be cost free in terms of time and energy consumption. 비유하자면, cognitivism은 뇌에서 벌어지고 있는 일을 컴퓨터 속의 CPU처럼 physical real world를 매우 간단하게 간주한다. 즉, real world를 physical dimension을 제외하고 하나의 추상적인 symbol로 만들어버린다. 이렇게 symbol단위로 표현하는 철학을 가지기 때문에 Cognitive science가 Computer science와 결합이 잘된다.     

### 2. Recursion
When such a symbol system, comprising arbitrary shapes of tokens (Harnad, 1992), is provided with **recursive functionality** for the tokens' operations, it achieves compositionality with an infinite rage of expressions. 이론상 무한대의 recursived symbols들이 존재할 수 있다.

[Chomsky의 인지과학] <br>
Noam Chomsky, famous for his revolutionary ideas on generative grammar in linguistics, has advocated that `recursion is a uniquely human cognitive competency`. Recursion은 사람 고유의 능력이다. 우리의 뇌가 recursion을 잘하도록 설계되어있다. 즉, 자연 구조는 recursive하게 구조화되어 있다고 볼 수 있다. 사람은 자연에 법칙에 따라 진화해왔기 때문에..

Chomsky and colleagues proposed that the human brain might host two distinct cognitive competencies:
  * faculty of language in a narrow sense (***FLN***): involves **only recursion** and regarded as a uniquely human aspect of language. FLN is thought to generate internal representations by utilizing syntactic rules and mapping them to a sensory-motor interface via the phonological system as well as to the conceptual-intentional interface via the semantic system
  * faculty of language in a broad sense (***FLB***): comprises a **sensory-motor system**, a **conceptual-intentional system**, and the **computational mechanisms** for recursion allowing for an infinite range of expressions from a finite set of elements

[실험] Chomsky admit that some animals (ex. Chimps) can exhibit certain recursion-like behaviors with training (ex. cup-nesting). They found that performance differed by species as well as among individuals. Cup-nesting 실험에서 하등동물의 nesting 횟수 (=depth of recursion)는 사람에 비하면 매우 제한적이었다. 사람의 경우는 물리적 조건(시간)만 주어진다면 무한대의 nesting을 할 수있다. 이번 실험을 통해 Chomsky와 colleagues는 추정하였다: `the human brain might be uniquely endowed with the FLN component` that enables infinite recursion in the generation of various cognitive behaviors including language.

[FLN?] Then, what is the core mechanism of FLN? It seems to be a recursive call of logical rules. 인간은 '사고'할 때, recursive하게 한다. 숫자를 셀 때도 그렇고, 셈을 할 때도 그렇고, 언어를 만들어 낼때도 그렇다. (In the recursive structure of sentences, clauses nest inside of other clasues, and in sentence generation the recursive substitution of one of the context free grammar rules for each variable could generate sentences of infinite length after starting with the symbol "S")

[Chomsky' View] Chomsky's crucial argument is that the core aspect of recursion is ~~not a matter of what has been learned or developed over a lifetime~~ `but what has been implemented as an innate function in the faculty of language in a narrow sense (FLN)`. In their view, what is to be learned or developed are the **interfaces**:
  * from this core apspect of recursion ability
  * to the sensory-motor systems or semantic systems in the faculty of language in a broad sense (FLB)

They assert that the unique existence of this core resursive aspect of FLN is an **innate component** that positions human cognitive capability at the top of the hierarchy of living systems.

[논쟁의 여지] <br>
하지만, 그들의 주장/견해는 논쟁(contentious)을 불러일으킬만하다. 그들의 주장대로라면, 훈련만 잘 받는다면, 무한대의 recursion을 할 수 있는 능력을 가질 수 있단 말인가? 예를 들어 무한대 길이의 방정식을 암산으로 풀기. 무한대 길이의 글 인식/생성하기. 하지만 일상생활에서 infinite recursion이 필요할까?
  * It is not realistic to assume that we humans perform infinite recursions in everyday life. However, Chomsky and colleagues see this not as a problem of FLN itself, but as a problem of external constraints (e.g., a limitation in working memory size in FLB in remembering currently generated word sequences) or physical time constraints that hamper performing infinite recursions in FLN.
  * Are symbols actually manipulated recursively somewhere in our heads when counting numbers or generating/recognizing sentences? 즉, 우리는 사물을 인식할 때 항상 recursive하게 인식하는가? 아니다. 예를 들어, 사과 3개가 있다면 우리는 visual하게 한꺼번에 몇 개인지 인식할 수 있다. 만약에 수십개가 있다면 하나하나 셀 수밖에 (recursive counting) 없을 것이다. 언어를 인지할 때 우리는 보통 (모국어라면) 문법을 하나하나 따지지 않고 자동적으로 하나의 구조자체로 한꺼번에 인식한다. 물론 복잡한 글귀같은 경우는 문법을 하나하나 따지면서 (recursive reading) 이해해야 할 것이다.

The notion of being infinite levels of recursons in FLN might apply only rarely to human cognition. In everyday life, it seems unlikely that an infinite range of expressions would be used.

그래도 many cognitive behavior들은 일상속에서 무한 대는 아니더라도 적당한 레벨의 composition 또는 recursion의 정보를 처리한다. 그렇다고 너무 심플한 레벨이면 조금 복잡한 behavior를 할 수 없다. some cognitive behaviors require some level of manipulation of internal knowledge about the world, yet does not involve infinite complexity. How is such processing done?
  * Use the core recursive component of calling logical rules in FLN under the **limitation of finite levels of recursions**.
  * Assume **subrecursive funtions** embedded in analogical processes rather than logical operations in FLB, that can mimic recursive operations for finite levels.

[정리 및 의문점] <br>
Cognitivism takes the former possibility with its strong conviction that the core aspect of cognition should reside in symbol representation and a manipulation framework. But, if we assume that symbols comprising arbitrary shapes of tokens convey the richness of meaning and context? 과연 symbol representation으로 meaning을 충분히 설명할 수 있는가? 예를 들어 사과를 "color-is-RED", "shape-is-SPHERE" 등으로 표현하면 사과의 의미를 충분히 표현한 것인가? 무한대의 recursive한 symbol은 의미를 풍부하게 다 표현할 수 있겠지만, Cognitivism은 인간은 위와 같이 limitation of finite levels of recursions을 가진다고 가정한다. 무엇보다 real world는 continuous하다. 반면, symbol은 discrete하다.




## Summary with My View



## Reference
> This chapter is a part of 'Exploring Robotic Minds' written by Jun Tani. I wrote this summary while taking his class, 'EE817-Deep Learning and Dynamic Neural Network Models'.
