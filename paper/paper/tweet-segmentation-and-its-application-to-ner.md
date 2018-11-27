# Tweet Segmentation and Its Application to Named Entity Recognition

* My note for this paper is in [**here**](https://onedrive.live.com/edit.aspx?cid=2ba5907d25ab4f59&page=view&resid=2BA5907D25AB4F59!205&parId=2BA5907D25AB4F59!193&app=Word)
* My presentation for this paper is in [**here**](https://onedrive.live.com/view.aspx?cid=2ba5907d25ab4f59&page=view&resid=2BA5907D25AB4F59!204&parId=2BA5907D25AB4F59!193&app=PowerPoint)

## Summary
* Noise(e.g. 글쓰는 스타일, 언어유희, 신세대 언어, 줄여쓰기, ..)하고 short한 특징을 가지는 tweet을 의미적으로 segmentation 잘하는 HybridSeg 알고리즘 설계
* Named entity, event detection, summarization 등과 같은 NLP에 이용하기 위한 첫 단추로 text segmentation이 활용될 수 있음

_HybridSeg 알고리즘_
* 외부 지식 정보를 사용하여 높은 성능의 segmentation을 잘 하는 HybridSeg 알고리즘 설계
   * global context (i.e. wikepedia와 같은 일반적인 정보)
   * local context (i.e. tweet와 같은 유니크(최신) 정보)
      * 공적인 기관, 정보, 회사 계정용 tweet (신뢰성이 높은 데이터임)
      * 같은 주제를 공유하는 tweet (e.g. 특정 시간대 tweet)
* 최적의 segment를 구하는 문제를 점수(stickiness score)를 최대화하는 최적화 문제로 바꿈 (가장 높은 점수를 가지는 segment s들을 출력)
   * L(s) - segment length normalization : 긴 길이의 segment는 더 깊은 의미를 가지므로 높은 점수를 얻음
   * Q(s) - wikipedia 활용 : 해당 segment가 wiki상에서 hyperlink를 많이 가지면 높은 점수를 얻음
   * SCP(s) - symmetric conditional probability 
   * Pr(s) (핵심) 
      * local context 
         * (공적인 tweet) pre-trained NER 모델을 사용함 -> NER모델이 segment를 인식했냐 안했냐..
         * (시간별/주제별 tweet) n-gram뿐만 아니라 sub-n-gram까지 사용 -> (sub) n-gram language model 사용
      * & global context에 의해 결정
      
      
