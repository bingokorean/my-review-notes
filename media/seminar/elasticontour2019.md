# Elastic On Tour

서울, 2019년 2월 22일

Nori 형태소 분석기 개발을 하고 있는데 Nori 개발자인 Jim이 한국에 온다는 소식에 세미나를 참가하게 되었다. 인사만 하려고 했으나 부스에서 친절하게 맞이해주고 질문에 답변해주셔서 매우 뜻깊은 시간을 보냈다. 추가로, 엘라스틱서치 CEO와 개발자들을 만나고 좋았고 엘라스틱서치의 장점과 기능을 한 번에 알게 되어서 좋았다. 무엇보다 점심이 매우 맛있었다. :smiley:

# 노리

### Nori, the genesis

* Mecab 
   * An open source text segmentation library
   * Language agnostic
   * Dictionaries for Japanese (ChaSen, IPADIC, ...)
* Lucene Kuromoji Analyzer
   * Based on Mecab and IPADIC
   * Implements the segmentatino part of the MeCab library in Java
   * Viterbi
* mecab-ko-dic
   * Created by Yongwoon Lee and Yungho Yu
   * Apache 2 License
   * A morphological dictionary for Korean language using Mecab
   * More than 800,000 entries
   * 3815 left ids, 2690 right ids (connection costs 3815*2690=10,262,350)
   * Used by Seunjeon
   * 200MB uncompressed

### Nori, Binary Dictionary

* Finite State Transducer (FST)
   * Prefix and infix compression for Hangul and Hanja
   * UTF-16 encoding
   * 5.4MB
* Connections costs
   * 3815 left ids, 2690 right ids (connection costs 3815*2690=10,262,350)
   * One short (16 bits) per cell
   * 20MB loaded in a direct byte buffer outside of the heap
* Feature encoding
   * Custom binary encoding
   * 9 bytes per entry (7MB total)

### Nori, morphological analysis

* Viterbi algorithm
   * Lattice
      * Find the best possible segmentation
      * Backtrace when only one path is alive


# 엘라스틱

엘라스틱이 적용된 아키텍쳐이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/arc1.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/arc2.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/arc3.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/arc4.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/arc5.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/arc6.PNG" width="80%" height="80%"></p>

엘라스틱의 장점이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/es1.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/es2.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/es3.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/es4.PNG" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/gritmind/review/blob/master/media/seminar/images/es5.PNG" width="80%" height="80%"></p>
