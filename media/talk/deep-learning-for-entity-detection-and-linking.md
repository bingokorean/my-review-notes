# Deep learning for entity detection and linking

2017.05.26 <br>
전북대학교, 나승훈

## Review
The review slides for this talk is [**here**](https://1drv.ms/p/s!AllPqyV9kKUrglHw8WgrzKxssKIg).

## Summary
* CRF는 sequential labeling task에 많이 사용되고 보통 바로 이전 (t-1)의 output만 참조함. 그 이상되면 모델이 굉장히 복잡해짐.
* CNN이 RNN을 approximation할 수 있음. 장점은 속도가 매우 빠름.
* Pointer network는 input word들을 잘 조합해주기 때문에 Summarization에 많이 사용. (새로운 word를 예측해야 한다면 사용하면 안됨.)
* CRF를 쓰지않고 RNN 하나로만 output dependency를 표현할 수 있음. (-> Language Model) 성능은 아마 더 떨어질 것임.
* char-embedding은 unknown word 처리에 유용함. NER의 경우 미등록 문제가 많음!.
* Event Detection은 미리 정해진 Event들의 trigger(ex.주로 동사)를 찾는 일임. Event Extraction은 trigger와 함께 그의 argument(ex.동사의 주어)까지 찾는 일임. 
