# Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems

My review note for this paper is in [here (presentation file)](https://1drv.ms/p/s!AllPqyV9kKUrj2HLbrsw1oSDQt5W)

## Summary
* feed-forward에 attention을 얹어서 long-term memory 문제 (e.g. addition, multiplication; temporal integration)를 해결.
* 장점은 length가 매우 긴 sequence나 sequence length variation이 큰 데이터셋에도 잘 됨.
* sequentially하게 동작하는 RNN에 비해 feed-forward with attention은 (sequential문제를 품에도 불구하고) parallelize (병렬화)를 할 수 있음 (계산 효율성).
* window based feed-forward도 long-term memory 문제를 해결할 수 있지만, feed-forward with attention과의 차이점은 시간마다 메모리 가중치를 공유하느냐에 달려있고 이는 temporal integration 능력에 영향을 미침. 즉, sequence length 길이에 관계없이 temporal integration을 잘 할 수 있음.
* 한계점은 long-term memory 문제에서 time order를 중요시하는 sequential correlation은 잘 이해하지 못함. 따라서, NER과 같은 문제에는 적합하지 않지만, 문서 분류 문제에서는 잘 맞을 수 있음. 

