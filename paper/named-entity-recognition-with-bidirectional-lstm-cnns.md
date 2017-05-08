# [Named Entity Recognition with bidirectional LSTM-CNNs](https://arxiv.org/pdf/1511.08308.pdf) (2015.11)

# Review note
The parts of Model & Evaluation with some figures and tables are illustrated in [[PPT Slides]](https://1drv.ms/p/s!AllPqyV9kKUrgkmsSb76RJbzJSor)

## Keyword
* hybrid bidirectional LSTM and CNN architecture
* encoding partial lexicon matches

## Review
The key points are using a hybrid of bi-directional LSTMs and CNNs that learns both character- and word-level features and using lexicon features with partial matching. The results are all state-of-the-art at CoNLL-2003 and OntoNotes datasets. The useful knowledge from this paper is that 'word embeddings trained on in-domain text may perform better' and 'deep learning models seem to get a better performance as data is bigger by comparing CoNLL with OntoNotes, which has larger dataset than CoNLL' and 'Capitalization features seems good features for NER task'

---

# Summary note

## Content
* Abstract
* Introduction
* Model
* Evaluation
* Results and Discussion
* Related Research
* Conclusion


## Summary

### 0. Abstract
* NER is challenging task because it requires heavy feature engineering and lexicons resources and rich enity linking information. (엔티티 링킹 정보도 feature로도 사용될 수도 있다.)
* We present a novel neural network architecture that automatically detects word- and character-level features using a **hybrid bidirectional LSTM and CNN architecture**, eliminating the need for most feature engineering.  
* We also propose a novel method of **encoding partial lexicon matches** in neural networks
* Given only tokenized text and publicly available word embeddings, our system is competitive on the CoNLL-2003 dataset and surpasses the previous state-of-the-art performance on the OntoNotes 5.0 dataset by 2.13 F1. By using two lexicons constructed form publicly-available sources, we establish new state of the art performance with an F1 score of 91.62 on CoNLL-2003 and 86.28 on OntoNotes.

> 딥러닝 모델의 철학이 들어간다. hand-craft feature engineering을 안하고 모델이 자동적으로 데이터에 의해 좋은 feature를 만들도록 설계한다.


### 1. Introduction
* CRF, SVM, Perceptron Models
  * hand-crafted features
  * need large human efforts
  * not sophisticated
* Feed-forward Neural Networks
  * word embedding trained by unsupervised learning
  * little feature engineering
  * restricts the use of context to a fixed sized window around each word
  * discards useful long-distance relations between words
  * can't exploit character level features such as prefix and suffix which could be useful with **rare words** where word embeddings are poorly trained
* Recurrent Neural Networks (RNN, LSTM, bi-directional LSTM)
  * allowing highly long-distance dependencies
  * taking into acount an effectively infinite amount of context (on both sides of a word)
  *  need large computational power

`For modeling character-level information or for extracting character-level feature (below table)`,

| LSTM | CNN |
|-|-|
| Including context information | Cheap computation |
| Expensive computation | Excluding context info, because of local-invariant property |

Preliminary evaluation shows that LSTM does not perform significantly better than CNNs while being more computationally expensive to train.

> character-level feature을 CNN으로 만드는 것이 이 논문의 핵심의 중의 하나임.

  * We present a hybrid model of bi-directional LSTMs and CNNs that learns both character- and word-level features.
  * We propose a new lexicon encoding scheme and matching algorithm that can make use of partial matches.


### 2. Model
[[PPT Slides]](https://1drv.ms/p/s!AllPqyV9kKUrgkmsSb76RJbzJSor) 참조

> 여기서는 transition score 역시 학습해야될 파라미터로 지정하고 있는데, 구지 데이터셋에서 학습할 필요가 있을까? 그냥 corpus에서 빈도수를 측정해서 이를 score로 사용하는게 더 좋지 않을까?

### 3. Evaluation
[[PPT Slides]](https://1drv.ms/p/s!AllPqyV9kKUrgkmsSb76RJbzJSor) 참조
> 이 논문에서는 실험적인 좌중우돌? 이야기 많이 있어서 좋다. 기억해두면 나중에 실험할 떄 많은 도움이 될 듯하다.

* For each experiment, we report the average and standard deviation of **10 successful trials**. (신뢰성을 좀 더 높일 수 있음.)
* Data Preprocessing (-> 전처리 최소화하였음.)
   * All digit sequences are replaced by a single '0'
   * Before training, we group sentences by word length into mini-batches and shuffle them. (계산 빨리하기 위해)
   * In addition, for the OntoNotes dataset, in order to handle the Date, Time, Money, Percent, Quantity, Ordinal, and Cardinal named entity tags, we split tokens before and after every digit.

### 4. Results and Discussion
[[PPT Slides]](https://1drv.ms/p/s!AllPqyV9kKUrgkmsSb76RJbzJSor) 참조
* Our best models have surpassed the previous highest reported F1 scores for both CoNLL-2003 and OntoNotes

### 5. Related Research
* CRF, SVM, perceptron models
  * is heavily dependent on feature engineering
  * (Ratinov and Roth (2009)) Broun-cluster-like word representations, non-local features, a gazetteer extracted from Wikipedia (F1:90.80 to CoNLL-2003)
  * (Lin and Wu (2009)) without a gazetteer, instead using phrase features obtained by performing k-means clustering over a private database of search engine query logs. (F1: better than 90.80)
  * (Passos (2014)) only public data by training phrase vectors in their lexicon-infused skip-gram model (F1:silmilar to Lin and Wu)
  * (Suzuki et al (2011)) to combat the problem of sparse features, employed large-scale unlablelled data to perform feature reduction (F1:91.02, which is the current state of the art for systems without external knolwedge)
  * (Durrett and Klein (2014)) combined coreference resolution, entity linking and NER into a single CRF model and added cross-task interaction factors (F1:state of the art to OntoNotes, not evaluate on CoNLL-2003)
  * (Luo (2015)) by training a joint model over the NER and entity linking tasks (F1: state of the art to CoNLL-2003)


* Neural Networks
  * (Petasis (2000)) a feed-forward neural network with one hidden layer. Only uses POS tag and gazetteer tags for each word, with no word embeddings.
  * (Hammerton (2003)) a single-direction LSTM network. Uses a combination of word vectors trained using self-organizing maps and context vectors obtained using principle component analysis.
  * (Collobert el al. (2011b)) SENNA, which employs a deep FFNN and word embeddings on POS tagging, chunking, NER, and SRL.
     * `We build on their approach, sharing the word embeddings, feature encoding method, and objective functions.`
  * (Santos (2015)) CharWNN network = { Collobert + character-level CNNs }, for Spanish, Portuguese NER.
     * `We have successfully incorporated character-level CNNs into out model for Enligsh NER.`
  * (Huang (2015)) a BLSTM for the POS tagging, chunking, NER, but use heavy feature engineering insted of using a CNN to automatically extract character-level features.
  * (Labeau (2015)) a BRNN with character-level CNNs, for German POS tagging.
     * `Our model differs in that we use the more powerful LSTM unit and word embedding, which is much more important in NER than in POS tagging`
  * (Ling (2015)) use both word- and character-level BLSTMs to establish the current state of the art for English POS tagging.
     * While using BLSTMs instead of CNNs allows extraction of more sophisticated character-level features, we found in preliminary experiments that for NER, it did not perform significantly better than CNNs and was substantially more computationally expensive to train.

### 6. Conclusion
* { A bidirectional LSTM + character-level CNN + dropout } with little feature engineering gets state-of-the-art results
* best score on NER dataset (CoNLL-2003, *OntoNotes*), suggesting that our model is capable of learning complex relationships from large amounts of dataset
* Partial matching lexicon algorithm suggests that performance could be further imporved through more flexible application of existing lexicons
* Word embeddings suggests that the domain of training data is as important as the training algorithm.
* 특히, lexicon과 embedding은 좀 더 많은 연구가 필요.



## Reference
Chiu, Jason PC, and Eric Nichols. "Named entity recognition with bidirectional LSTM-CNNs." arXiv preprint arXiv:1511.08308 (2015).
