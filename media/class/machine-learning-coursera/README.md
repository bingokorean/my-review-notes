# Machine Learning

Coursera / Stanford / Andrew Ng <br>
2015.08 ~ 10 <br>
I has successfully completed this course at 2015.11.17 [[certificate]](https://github.com/gritmind/review/blob/master/media/class/machine-learning-coursera/certificate.pdf) <br>


## Contents
* Week1. Introduction / Linear Regression with One Variable [[note]](https://1drv.ms/w/s!AllPqyV9kKUrhC1Tuaq4TIgOezWy)
* Week2. Linear Regression with Multiple Variables / Octave/Matlab Tutorial [[note]](https://1drv.ms/w/s!AllPqyV9kKUrhC4_nRMMtR-JurqM)
* Week3. Logistic Regression / Regularization [[note]](https://1drv.ms/w/s!AllPqyV9kKUrhDpL4wwNVl9yrF-x)
* Week4. Neural Networks: Representation [[note]](https://1drv.ms/w/s!AllPqyV9kKUrhFKWcdnz2_WNLbK1)
* Week5. Neural Networks: Learning [[note]](https://1drv.ms/w/s!AllPqyV9kKUrhFk9ZEv8tzfBYWFM)
* Week6. Advice for Applying Machine Learning / Machine Learning System Design [[note]](https://1drv.ms/w/s!AllPqyV9kKUrhGH9hALxltRYZvCG)
* Week7. Support Vector Machines [[note](https://1drv.ms/w/s!AllPqyV9kKUrj11mEgwhkkwMp6_C)]
* Week8. Unsupervised Learning [[note](https://1drv.ms/w/s!AllPqyV9kKUrj2RwgChGdjLW3ABW)]
* Week9. Anomaly Detection & Recommender System [[note](https://1drv.ms/w/s!AllPqyV9kKUrkFDeNRj5Zpm1tLln)]
* Week10. Large Scale Machine Learning [[note](https://1drv.ms/w/s!AllPqyV9kKUrkFtcA1F2CmQY6DxL)]
* Week11. Application Example: Photo OCR [[note](https://1drv.ms/w/s!AllPqyV9kKUrkFyu6S_wnB7ci_-G)]


### Programming Assignments
* [PA1](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/1-linear-regression). Linear Regression [[*matlab-notebook*](https://1drv.ms/w/s!AllPqyV9kKUrhDsHH0NRJdbzudzc)]
* [PA2](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/2-logistic-regression). Logistic Regression [[*matlab-notebook*](https://1drv.ms/w/s!AllPqyV9kKUrhDyzsn-a_tK_rJvd)]
* [PA3](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/3-multi-class-classification-and-neural-networks). Multi-class Classification and Neural Networks [[*matlab-notebook*]](https://1drv.ms/w/s!AllPqyV9kKUrhFFmDyBqz8kDBPic)
* [PA4](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/4-neural-networks-learning). Neural Networks Learning [[*matlab-notebook*]](https://onedrive.live.com/view.aspx?resid=2BA5907D25AB4F59!600&ithint=file%2cdocx&app=Word&authkey=!ANzlLu3om1tLAgE)
* [PA5](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/5-Regularized-Linear-Regression-and-Bias-vs-Variance). Regularized Linear Regression and Bias vs. Variance [[*matlab-notebook*]](https://1drv.ms/w/s!AllPqyV9kKUrhGDkzE1z7koyqZii)
* [PA6](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/6-support-vector-machines). Support Vector Machines [[*matlab-notebook*](https://1drv.ms/w/s!AllPqyV9kKUrj2U13U88HZpbbBDv)]
* [PA7](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/7-PCA-and-K-means). Principle Component Analysis and K-Means Clustering [[*matlab-notebook*](https://1drv.ms/w/s!AllPqyV9kKUrj2hWHCEfERwBVfP7)]
* [PA8](https://github.com/gritmind/review/tree/master/media/class/machine-learning-coursera/assignments/8-anomaly-and-recommender). Anomaly Detection and Recommender Systems [[*matlab-notebook*](https://1drv.ms/w/s!AllPqyV9kKUrkFFhy_tUeYt9ZiqK)]

<br>

## Comments

It was the first course for me to learn machine learning.  As the beginner, I could take the course without big difficulty, but MATLAB programming assignments were a little challenging. 

I could learn the various types of machine learning algorithms through some real-world examples such as house price, spam classification and learn more deeply or interestingly through MATLAB programming assignments which require the way of **vectorial implementation** specialized in parallel hardware rather than simple for-loop logic, especially when building gradient descent and forward propagation algorithms. 

Given the starter code of the program, I could concentrate only on the algorithm function core part, which reduces some redundant time. 

I have learned the **basic machine learning models** such as Linear or Logistic Regression, Neural Network, Clustering, Support Vector Machine, Collaborative Filtering and other various data mining techniques such as anomaly detection, dimensionality reduction, and so on.

Lastly, since the course consists of the basic concepts, I feel I need to have more deep inside from other courses.

<br>


## Summary

Recommender System

* 개요
   * 추천 시스템 알고리즘(collaborative fitlering)은 신경망과 같이 자동적으로 자질을 추출해준다.
   * 문제는 movie/user 행렬에서 주어진 rating을 가지고 rating이 되지 않은 빈칸을 채우는 일이다.
   * 일반적인 감독학습과 다른점은, 어떤 movie, 어떤 user를 토대로 rating을 예측하는 것이다. x 측면이 movie와 user로 2가지가 있다는 점이다. 사실, 이러한 측면으로 collaborative filtering이 가능하다.
* 모델 개수 관점 구분
   * 통합적인 모델 1개로 구성해보자.
      * X = ((장르, 배우, 감독), (사용자ID, 국적, 나이, 남/여)), y = rating
      * 이렇게 한꺼번에 parameter들을 뭉쳐서 학습을 한다면?
      * movie와 user의 feature를 모두 manually 설계해야 한다.
      * 통합적이므로 user 개개의 모델을 만들 수 없다. (비효울적으로 느껴진다)
      * 위의 데이터 특징으로 user들 사이에 상관성이 있다는 것이다. user를 분리시키는 것이 현명하다.
   * 모델 n개로 구성해보자. (user 개수만큼)
      * X = (장르, 배우, 감독), y = rating
      * user가 기준이 되어야 한다. 우리는 user에게 서비스를 제공해야 하기 때문이다.
      * user 1명당 모델을 1개 가진다. 따라서, user들의 성향을 구분할 수 있다.
      * 전체 모델에서 user 기준으로 매우 세세하게 모델링한 것과 같다.
      * 추천 시스템에 일반적으로 선택되는 모델 아키텍쳐이다. (아래 content-based, colloborative 모두 포함)
* 자질 설계 관점 구분
   * Content-based 추천 시스템
      * movie 축의 feature를 manually 설계한다.
      * 현실적으로 매우 많은 영화들의 feature를 일일이 정의하기 어렵다. 예를 들어, 모든 영화들을 구체적으로 분석해서 장르별로 degree를 부여하는 것은 쉽지 않다.
      * feature 차원이 하나 늘어날 때마다 비용이 매우 많이 든다.
      * movie 개수를 학습 데이터 개수라 볼 수 있다. movie 개수가 많아지면 각 user 모델의 성능은 높아진다.
   * Collaborative Filtering
      * feature learning을 하는 특징을 가진다.
      * feature가 무엇인지 글자로 정의할 수 없다. 그냥 수치이다. feature vector 차원만 정의할 뿐이다.
      * movie feature vector는 영화에 대한 장르 degree의 묶음으로 생각할 수 있다.
      * user feature vector는 user preferences(user들의 영화 장르에 대한 선호도)로 개념화시킬 수 있다.
      * 기준은 똑같이 n개의 user 모델이 있다는 것을 기억.
      * collaborative filtering 알고리즘의 한 가지 제한 사항으로 각 user가 다수 영화에 rating 점수가 매겨져 있어야 하고, 각 movie가 다수의 user들에게 rating 점수가 매겨져 있어야 한다. (cold-start 문제가 심할 듯)
      * 학습이 완료된 후에 user 모델 사이는 완전히 독립이다. 
      * 근데... 뭐가 협력적이냐..?
         * 어떤 사람이 어떤 movie에 rating을 하면 자신의 user feature뿐만 아니라 해당 movie feature에 반영된다. 업데이트되는 movie feature는 모든 user 모델이 사용한다. 그리고 후에 업데이트되는 자신의 user feature도 모든 movie feature에 영향을 준다.
         * 비슷한 성향의 user 들이 비슷한 user vector를 가지게 될 것이고, 비슷한 장르의 movie 들이 비슷한 movie vector를 가지게 될 것이다. 
         * 생각해볼 점 -> 어떤 user vector와 어떤 movie vector의 유사도가 높다는 의미는 무엇일까?
   * Matrix Factorization
      * 선형대수학을 사용해서 feature들을 추출할 수 있다. 위에서는 사람이 manual하게 결정. CF는 feature dim 개수만 사람이 결정.
      * feature는 latent variables와 같다. 즉, movie feature vector는 movie latent feature vector, user feature vector는 user latent feature vector가 된다.

