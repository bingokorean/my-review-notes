# Sequence to Sequence Learning with Neural Networks (2014)

* Organized note for this paper [**[note]**](https://1drv.ms/w/s!AllPqyV9kKUrg2er-V6RvUwzJa43)
* Presentation for this paper [**[ppt]**](https://1drv.ms/p/s!AllPqyV9kKUrg2j_90qqj6OWWAEF)

## Summary
* '다:다' 입출력이 가능한 모델 (입출력 길이달라도 상관없음)
* encoder로부터 얻은 h 벡터를 사용해 sentence embedding이 가능
   * 단어 순서 정보는 sent embedding에 영향을 줌
   * 능동태/수동태 정보는 sent embedding에 그다지 영향을 안줌
* **(KEY CONSTRIBUTION)** source sentence의 단어 순서를 바꿈으로써 성능향상 (minimal time lag 최소화로 인한 최적화 향상)
* mini-batch 시에 길이가 비슷한 것끼리 그룹핑한 후 학습하면 최적화에 도움
* 한정된 VOCA size를 가지는 (size: 80,000 그외엔 unknown) LSTM 모델이 기존의 SMT 모델 (VOCA가 한정되지 않음)보다 성능 더 잘나옴.
* 긴 문장에 대해서도 LSTM이 좋은 성능을 가짐. (제한된 LSTM 메모리때문에 잘 안될 줄 알았으나 잘되었음) 
