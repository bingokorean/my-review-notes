# Learning to Rank: From Theory to Production

Malvina Josephidou & Diego Ceccarelli <br>
Bloomberg <br>
Presented at Activate 2018 (THE SEARCH AND AI CONFERENCE)

This talk, by the engineers at Bloomberg who built this functionality into Solr in the first place, is a war story of how the company's real-time, low-latency news search engine was tamed to learn how to rank

* See. [Youtube](https://www.youtube.com/watch?time_continue=16&v=eMuepJpjUjI). <br>
* [Annotated Slides](https://github.com/gritmind/review/blob/master/media/talk/learning_to_rank_bloomberg/learningtorankfromtheorytoproduction_gritmind.pdf), annotated by me.

## Review

* 기존 시스템(solr)에서 랭킹을 더 잘하기 위해 추가적인 모듈, Re-ranking model을 붙임.
* 정확도뿐만 아니라 특히, 속도 향상을 위해 굉장히 공을 들임.
* Re-ranking은 복잡한 모델과 feature를 사용하므로 입력 데이터수를 최대한 줄여야 함.
   * 앞단에 있는 일반적인 랭킹의 결과(Top-x retrieval)를 사용.
   * 속도 향상을 위해서 static한 feature는 미리 계산해 놓는 것이 좋음.
   * Grouping을 통해서 한 번 더 입력 데이터수를 줄임.

현업에서 속도를 유지하면서 좋은 모델과 좋은 자질을 사용하는 일은 매우 어려운 것 같다.