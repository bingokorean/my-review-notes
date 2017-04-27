# [Neural Architectures for Named Entity Recognition (2016)](https://arxiv.org/pdf/1603.01360.pdf)

# Review note
The figures and tables in this paper are arranged in this [[PPT]](https://1drv.ms/p/s!AllPqyV9kKUrgju5EEGUgHB7-3gy). 

## Keyword
* hand-crafted features & domain-specific knolwedge & resources
* neural architecture (Bi-LSTM)
* conditional random field (CRF)
* character-level word representation: orthographic or morphological evidence
* dropout

## Review
For solving NER task, the problem is small lableded data and such entities have various kinds of contexts. The crucial is we have small labeled data, so generalization is difficult. To overcome these, the previous works tried to use unsupervisded learning and hand-crafted features, requiring domain-specific knowledge and resources. However, designing hand-crafted features is coslty. In this paper, they present new neural architecture, that is Bi-LSTM-CRF, which use both supervised and unsupervised, and that do not require domain-specific knolwedge and resources. In detail, for incorporating dependencies they add CRF layer on the top. For more expressing orthographic or morphological evidence, they use character-level word representation. Lastly, for getting good generalization among various kinds of word representations, they add dropout layer. This model has state-of-the-art score on CoNLL-2003 data, which is a representative NER data.      


---

# Summary note

## Content
* Abstract
* Introduction
* LSTM-CRF model
* ~~Transition-Based Chunking model~~
* Input Word Embeddings
* Experiments
* ~~Related Work~~
* Conclusion


## Summary

### 0. Abstract
Due to **small** supervised training corpora, the preivous model rely heavily on **hand-crafted** features and **domain-specific knowledge**. On the other hand, our model as **neural architectures (LSTM-CRF)** rely on character-level and word-level **word representaton** trained by unsupervised learning and **no language-specific knowledge** or resources.

| Previous Model | Our Model |
|---------|-----|
| hand-crafted features | features made by neural architectures |
| language-specific knolwedge or resources | **No** such knowledge or resources |
| only supervised learning | both supervised and unsupervised leearning |

> 자질을 설계하는데 있어 영어-도메인 지식이 전혀 필요없다는 것이 인상깊다.


### 1. Introduction
[문제 인식]
Named entity recognision (NER) is a **challenging** learning problem because in most languages and domains, there is only a **very small** amount of supervised training data available. Also, since there are **few contraints** on the kinds of words that can be names, generalizing from this small sample of data is difficult. `As a result, carefully constructed orthographic features and language-specific knolwedge resources such as gazetteers are widley used for solivng this task`. Unfortunately, language-specific resources and features are **costly** to develop in new languages and new domains, making NER a challenge to adapt.

> NER는 어려운 문제임 -> 적은 라벨 데이터 -> 복잡하고 다양한 패턴의 엔티티 정보를 모델이 습득하기엔 역부족 -> 따라서, 섬세하게 디자인되는 hand-craft 중심의 자질 설계가 필요 -> 그럴려면 도메인-지식이 필요 -> 하지만 그에 따른 비용감수가 필요 (문제점)  

[기존 연구]
**Unsupervised learning** from unnotated corpora offers an alternative strategy for obtaining better generalization from small amounts of supervision. However, even systems that have relied extensively on unsupervised features have used these to augment, rather than replace, hand-engineered features and specialized knowledge resources.

> 그래서 비지도 학습의 방법론들이 기존 연구에서 이루어졌으나, 그래도 그들은 여전히 hand-craft와 도메인-지식을 적극적으로 사용하고 있다.

[우리 연구]
In this paper, we present neural architectures for NER that use **no language-specific resources or features** beyond a small amount of supervised training data and unlabeled corpora. `Our models are designed to capture two intuitions`:
1. Since names often consist of multiple tokens, reasoning **jointly over tagging decisions** for each token is important. We use a bidirectional LSTM with a sequential conditional random layer above it (LSTM-CRF)
2. Token-level evidence for "being a name" includes both orthographic evidence and distributional evidence. To capture orthographic sensitivity, we use **character-based** word representation model and to capture distributional sensitivity, we combine these representations with **distributional** representations. Our word representations combine both of these, and **dropout** training is used to encourage the model to learn to trust both sources of ecidence. (무분별한 concatenation을 하기 때문에 dropout이 유용할듯...) [[PPT]](https://1drv.ms/p/s!AllPqyV9kKUrgju5EEGUgHB7-3gy)

> 우리 연구는 (언어) 도메인-지식을 전혀사용하지 않는다. 그리고 크게 2가지 컨셉을 가진다. 1.여러개의 단어들로 이뤄진 ENTITY를 잘 표현하기 위해 jointly over tagging decision을 한다. 2. 형태학적인 문자 특징을 잘 표현하기 위해 charactoer-based representation과 distibutional representation 사용, 위 2가지 컨셉을 잘 보면 모두 **데이터 특징** 을 잘표현한 거에 지나지 않는다. 따라서, 모델 설계든 자질 설계든 데이터 특징을 이해하는 것이 매우 중요하다.

### 2. LSTM-CRF model
[[PPT]](https://1drv.ms/p/s!AllPqyV9kKUrgju5EEGUgHB7-3gy)
### 2.1 LSTM
We use Two kinds of LSTMs such as the forward LSTM and the backward LSTM. `These are two distinct networks with different parameters`. As one unified model, it is referred to a bidirectional LSTM, simply concatenating both LSTMs.

#### 2.2 CRF Tagging Models
For solving POS tagging task, independent classification decisions are limiting when there are strong dependencies across output labels. Therefore, instead of modeling tagging decisions independently, we model them jointly using a conditional random field (CRF). How to train LSTM-CRF model? -> [[PPT]](https://1drv.ms/p/s!AllPqyV9kKUrgju5EEGUgHB7-3gy)

> 확률들의 sequence 가장 확률값이 높게 나와야 하므로 dynamic programming (ex. viterbi algorithm)을 사용한다.
 
#### 2.3 Parameterization and Training
The parameters in the LSTM-CRF:
  * Biagram compatibility scores A
  * Weights
  * Word Embeddings

#### 2.4 Tagging schemes
Sentences are usually represented in the IOB format (Inside, Outside, Beginning) where every token is labeled as B-label if the otken is the beginning of a named entity, I-label if it is inside a named entity but not the first token within the named entity, or O otherwise.

However we decide to use the IOBES tagging scheme, a variant of IOB commonly use for named entity recognition.IOBES encodes:
  * S-label: singleton entities
  * E-label: explicitly marks the end of named entities

> 이것 또한 데이터 특성 이해에 기반한 테크닉이다. 데이터에서 패턴이 보이면 최대한 활용해야 한다.

### ~~3. Transition-Based Chunking Model~~

### 4. Input Word Embeddings
[[PPT]](https://1drv.ms/p/s!AllPqyV9kKUrgju5EEGUgHB7-3gy) [전략1] Since many languages have **orthographic or morphological evidence** that something is a name (or not a name), we want representations that are sensitive to the spelling of words. We therefore use a model that constructs representations of words from representations of the ***characters***. [전략2] Secondly, names which may individually be quite varied, appear in **regular contexts** in large corpora. Therefore, we use ***embeddings*** learned from a large corpus that are sensitive to word order. [전략3] Finally, to prevent the models from depending on one representations or the other too strongly (=good generalization), we use ***dropout*** training.

#### 4.1 Charater-based models of words
Instead of hand-engineering prefix and suffix information about words, we use character-level word representations.
```diff
- hand-engineering prefix and suffix informatino about words
+ character-level word representations
```
Advantage of character-level word representations:
  * Useful for morphologically rich languages
  * To handle the out-of-vocabulary problem (ex. UNK embedding) for tasks liks part-of-speech tagging and laguage modeling or dependency parsing. (In word-level, we just use UNK embedding with a probability 0.5)

To make character-level word representations, we use LSTM. Why?

| CNN | LSTM |
|---------|-----|
| [+] no biased  | [-] It is biased for suffix (from forward LSTM) and prefix (from backward LSTM) |
| [-] position-invariant features (but text has position-dependent) | [+] |

> CNN과 LSTM 각각 장단점이 있는 것 같다. 아마 task에 따라 달라질 것 같다. 일종의 model hyper-parameter라고 생각해도 될 듯..

#### 4.2 Charater-based models of words
We use pretrained word embeddings using skip-n-gram, a variation of word2vec that accounts for word order. We use an embedding dimension of 100 for English, 64 for other languages, a minimum word frequency cutoff of 4, and a window size of 8.

#### 4.3 Dropout training
Initial experiments showed that character-level embeddings did not improve our overall performance when used in conjunction with pretrained word representations. `To encourage the model to depend on both representations`, we use **dropout** training, applying a dropout mask to the final embedding layer just before the input to the bidirectional LSTM. We observe a significant improvement in our model's performance after using dropout.

> Dropout으로 성능향상, 데이터가 작은 것이 이 task의 문제점이지만 dropout이 성능향상을 주었다.

### 5. Experiments

#### 5.1 Training (Only for LSTM-CRF Model)
* Backpropagation algorithm & stochastic gradient descent (SGD)
* Learning rate = 0.01
* Gradient clipping = 5.0
* SGC with gradient clipping is better than other advanced techniques such as Adadelta, Adam, which can have faster convergence.
* Single layer for each LSTM
* For each LSTM, 100 dimension
* Dropout rate = 0.5 (Using hihger rates negatively impacted our results, while smaller rates led to longer training time)

#### 5.2 Data Sets
* CoNLL-2002 (English NER, German NER)
* CoNLL-2003 (Dutch NER, Spanish NER)
* Four different types of named entities: locations, persons, organizations, and miscellaneous entities
* Although POS tags were made available for all datasets, we did not include them in our models.
* We did not perform any dataset preprocessing, apart from replacing every digit with a zero in the English NER dataset.

#### 5.3 Results
[[PPT]](https://1drv.ms/p/s!AllPqyV9kKUrgju5EEGUgHB7-3gy)

### ~~6. Related Work~~

### 7. Conclusion
* Our neural architecture provide the best NER results (compared with with models that use external resources)
* + CRF (using virtue of dependencies)
* Word representations (word embedidng + character-level) to capture morphological and orthographic information
* Dropout to prevent depeding too heavily on one representation class





## Reference
Lample, Guillaume, et al. "Neural architectures for named entity recognition." arXiv preprint arXiv:1603.01360 (2016).
