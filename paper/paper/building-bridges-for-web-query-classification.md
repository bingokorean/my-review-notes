# Building Bridges for Web Query Classification

* My note for this paper is in [**here**](https://onedrive.live.com/edit.aspx?cid=2ba5907d25ab4f59&page=view&resid=2BA5907D25AB4F59!208&parId=2BA5907D25AB4F59!193&app=Word)
* My presentation for this paper is in [**here**](https://onedrive.live.com/edit.aspx?cid=2ba5907d25ab4f59&page=view&resid=2BA5907D25AB4F59!207&parId=2BA5907D25AB4F59!193&app=Word)

## Summary
* 사용자에 의해 생성되는 web query를 이해하는 알고리즘 설계

_Aspect 1 Problem_
* web query는 short & ambiguous, multiple meaning, lack of training data과 같은 문제점 (aspect 1 문제점)을 가짐
* fixed target categories와 함께 topic classification 모델 설계
* 부족한 정보의 query를 확장하기 위해 검색 엔진 또는 ODP를 사용하여 query와 관련된 textual 정보와 categorical 정보를 획득; 이러한 정보들은 intermediate object라 불림
* imterdediate와 target cagegory를 연결하기 위해 두 가지 종류의 synonym-based classifier 사용 (최종적으로 ensemble해서 사용)
   * direct & extended matching (using wordnet) - high precision, low recall 문제점
   * statistical classifier (i.e. SVM) - high recall (기계학습 모델의 genearlization 능력을 활용)
   
_Aspect 2 Problem_
* web query는 meaning may evolve over time의 문제점이 있음 (위와 같은 경우 의미가 바뀌면 모든 classifier들을 다시 재학습해야 함)
* intermediate taxonomy를 bridge로 둬서 query와 target taxonomy를 연결함
* query와 intermediate taxonomy 사이, intermediate taxonomy와 target taxonomy 사이의 확률 기반 bridging classifier를 사용하여 similarity 정도를 측정하여 이를 토대로 가장 최적의 query와 target taxonomy를 찾음.
* intermediate taxonomy가 굉장히 많은 category를 가질 수 있는데 계산 복잡도를 줄이기 위해서 query와 irrelevant한 것들을 미리 제거할 수 있도록 feature selection과 비슷한 category selection (e.g. total probability, mutual information)을 실시 


