# Unit 2

## Summary

* Step 함수 (R 언어)
   * AIC를 기준으로 model simiplicity와 R2의 타협점을 가지는 변수들의 조합을 자동적으로 선택해준다. 
   * step함수으로부터 생성된 모델은 최소한 multicollinearity는 가지지 않는다.

* Multicollinearty Problem
   * insignificant 변수를 삭제 해야 함 (주의: 한 번에 한 개씩 삭제 요망)
   * 여러 개의 insignificant 변수들이 있는 경우, Multiple R-squared와 Adjusted R-squared 점수를 따져 가면서 효용성이 큰 변수들을 살리는 것이 좋음
   * 결국 모든 경우의 실험들을 다 해보고 가장 좋은 조합을 찾아야 하나, 현실적이지 않다면 외적인 전문 지식을 활용해서 효용성이 큰 변수들을 살리는 것이 좋음
   * 학습된 coefficient의 음양부호, 크기를 참고하여 기존에 알고 있는 정보와 일치하지 않으면 multicollinearity를 의심해볼 필요가 있음
   
* Add new variable
   * 해박한 도메인 지식 필요
   * 기존 변수들을 산술적으로 조합하여 새로운 변수를 만듦 (e.g. 빼기, 더하기, 곱하기, 나누기 등등)
   * 새로운 변수를 위해 하나의 모델을 만들 수도 있음. (e.g. cascade feature)

* 좋은 linear 모델이란?
   * out-of-sample의 성능인 R2 또는 RMSE가 높은 모델이 가장 좋음. (in-sample 성능인 multiple R2 또는 RMSE와 성능 비교)
   * in-sample로만 평가해보면.. (설명가능한 linear모델의 경우 학습 데이터만으로도 충분히 모델을 평가할 수 있음) 
      * multiple R-squared와 Adjusted R-squared의 점수가 높은 경우 strong model이라 간주함.
      * 추가로, 독립 변수들의 degree of significance가 모두 높은 경우
      * 위와 같은 조건을 만족하는 상황이라면 (비슷한 r 점수도 괜찮), 변수가 더 적을 수록 좋음 (심플한 모델일수록 좋음)
      
* linear 모델에서 좋은 변수란?
   * degree of significance가 높은 경우
   * coefficient 수치가 큰 경우. 즉, 더 큰 영향을 준다고 볼 수 있음
