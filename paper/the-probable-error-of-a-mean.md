# The Probable Error of a Mean

To easily grasp up the story of this paper, I refered to [Article (S. L. Zabell, 2008)](http://cda.mrs.umn.edu/~jongmink/Stat2611/s1.pdf) and [Summary](http://lhs.kennyiams.com/FilesAPStats/2nd%20Sem/23D%20students%20t%20model%20article%20summary.pdf)

My presentation for this talk is in [**here**](https://1drv.ms/p/s!AllPqyV9kKUrhFQePhgUf7iAr2xE)

## Summary
* small sample data로부터 large population을 추론할 수 있는 t-test의 초기 논문
* cost와 time때문에 small sample data밖에 가질 수 없는 상황에서 어떻게 large population을 추론할 수 있나?
* large sample data가 있으면(e.g. n>30) CLT(Central Limit Theorem)에 의해 population의 mean을 알 수 있음
* small sample data가 있으면 CLT를 적용하지 못하므로 population에 대한 아무런 정보를 얻을 수 없음 
* t 분포 등장
   * sampling error를 포함할 수 있는 statistic을 정의할 필요가 있음
   * sampling 분포의 평균을 추론하기 위해 sample 표준편차을 사용
   * z 분포에서 모집단의 표준편차를 sample의 표준편차로 교체 (일종의 근사치로 표현)
* population이 normal 분포라 가정
   * 모집단의 평균은 알지만, 표준편차은 모르는 경우 -> Chi-squared 분포 사용
   * 모집단의 평균은 모르고, 표준편차은 아는 경우 -> Standard Normal 분포 사용
   * 모집단의 평균과 표준편차을 모두 모르는 경우 -> T 분포 사용
   
