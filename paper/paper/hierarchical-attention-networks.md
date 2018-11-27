# Hierarchical Attention Networks for Document Classification

Review note for this paper is in [here (presentation file)](https://1drv.ms/p/s!AllPqyV9kKUrj19kiQX2d-oqTtjj).

## Summary
* Hierarchical Attention Network (RNN based) 문서 분류를 위한 새로운 신경망 제시
   * document에 hierarchical structure가 있다고 가정 -> Hierarchical (word가 모여 sentence, sentence가 모여 document)
   * 단어/문장 마다 중요도가 다름. 심지어 중요도가 단어/문장에 고정되어 있는 것이 아닌 문맥에 따라 변함 (highly context dependent) -> Attention mechanism
* (중요) attention mechanism의 입력은 단어 워드 임베딩이 아닌 단어 문맥 정보이다 (즉, GRU의 출력임) -> the importance of words or sentences is highly context dependent!!
* Pre-trained 워드 임베딩을 사용하는 것이 아닌, task-specific dataset으로 word2vec을 학습시킨 워드 임베딩을 사용
* attention mechanism으로 word visualization을 할 수 있음 -> visualize words carrying strong sentiment
## Reference
Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.
