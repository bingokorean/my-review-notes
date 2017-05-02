# 6. Support vector machines

# Review note

## Keyword
* hyperplane = decision boundary = separating line
* distance = perpendicular to the line
* margin
* support vector
* constrained optimization problem
* SMO algorithm
* inner product
* kernel trick


## Review
The goal of SVMs is to maximize margins of support vectors. The philosophy of SVMs is that the farther a data point, the more confident prediction is. Also, the one strength is to be beautifully able to use kernel trick because of inner product form. But, since there are many parameters, we must carefully tune them. I think SVMs consists of 3 steps rather than logistic regression consisting of 1 step.
  * First, to find the good separating line, meaning that find support vectors
  * Second, to maximize margins of suport vectors around the separating line
  * Third, to use kernel trick for better feature representation

The important figures and explanations are organized in [PPT Slides](https://1drv.ms/p/s!AllPqyV9kKUrgkXAkeTxXi7TmCak).

---

# Summary note

## Content
* Introducing support vector machines
* Using the SMO algorithm for optimization
* Using kernels to "transform" data
* Comparing support vector machines with other classifiers


## Summary

### 6.0 Intro
Support vector machines make good decisions for data points that are outside the training set. There are many implementations of support vector machines, but we focus on one of most popular implementations: the sequential minimal optimization (**SMO**) algorithm. After that, you'll see how to use something called **kernels** to extend SVMs to a larger number of datasets.

### 6.1 Separating data with the maximum margin
If data points are separated enough that you could draw a straight line, we say the data is **linearly separable**. The line used to separate the dataset is called a **separating hyperplane**. In 2D plots, it's just a line. In 3D plots, it's a plane. In N-D plots, it's a hyperplane. That is, the hyperplane is our **decision boundary**.  

We'd like to make our classifier in such a way that the farther a data point is from the decision boundary, the more confident we are about the prediction we've made. We'd like to find the point closest to the separating hyperplane and make sure this is as far waya from the separating line as possible. This is known as a **margin**. We want to have the greatest possible margine because we want our classifier to be as robust as possible. The points closest to the separating hyperplane are known as **support vectors**.

Now that we're trying to maximize the distance from the separating line to the support vectors, we need to find a way to optimize this problem.

> LR과 비교해보면 좋은 fitted line을 찾는 것은 같지만, SVM은 한 단계 더 나아가서 margin이 최대가 되는 fitted line을 찾는다.

### 6.2 Finding the maximum margin
[PPT Slide](https://1drv.ms/p/s!AllPqyV9kKUrgkXAkeTxXi7TmCak) 참조

### ~~6.3 Efficient optimization with the SMO algorithm~~

### ~~6.4 Speeding up optimization with the full Platt SMO~~

### 6.5 Using kernels for more complex data
[PPT Slide](https://1drv.ms/p/s!AllPqyV9kKUrgkXAkeTxXi7TmCak) 참조 <br>
When data is non-linear, our classifier cannot recognize that. Our classifier can only recognize greater than or less than 0. If we just plugged in our X and Y coordinates, we wouldn't get good results. Instead, we can recognize it by transforming the data from one feature space to another. Mathematicians like to call this **mapping from one feature space to another feature space**. Usually, this mapping goes from a lower-dimensional feature space to a higher-dimensional space. This mapping is done by a **kernel**. After making the substitution, we can go about solving this linear problem in high-dimensional space, which is equivalent to solving a non-linear problem in low-dimensional space.

> 'mapping from one feature space to another feature space' 이라는 말을 그냥 '또 다른 distance metric 사용'이라고 볼 수 있다고 저자가 설명하고 있다. 직감적으로 이해가 가능하다.

One great thing about the SVM optimization is that all operations can be written in terms of **inner products**. We can replace the inner product with a kernel is known as the **kernel trick** or kernel substation.

> SVM의 최정화 방정식에 inner product term이 있기 때문에 kernel을 사용하기가 유용하다. kernel 역시 inner product로 이뤄져있기 때문에, 그 자리에 kerner inner product로 교체만 해버리면 된다.

There is an optimum number of support vectors.

> Kernel에서도 parameter가 있음. sigma를 조절하면서 support vector의 개수를 조절할 수 있음. 모든 데이터를 support vector로 만들 수도 있음. 이것은 특정 데이터마다 다르므로 tuning할 필요가 있음.

The beauty of SVMs is that they classify things efficiently.
  * If you have too few support vectors, you may have a poor decision boundary.
  * If you have too many support vectors, you're using the whole dataset every time you classify something - that's called **k-Nearest Neighbors**.

> SVM와 KNN의 연결성: 모든 데이터가 Support vector일 때, SVM은 KNN이 된다.

### ~~6.6 Example: revisiting handwriting classification~~

### 6.7 Summary
* Support vector machines are a type of classifier
* They're called machines because they generate a binary decision; they're decision machines
* Support vectors have good generalization error: to be the best stock algorithm in unsupervised learning
* Support vector machines try to maximize margin by solving a quadratic optimization problem. In the past it's slow. But, not it's fast by using SVM algorithm, which allows fast training of SVMs by optimizing only two alphas at one time.
* Kernel trick map (nonlinear) data from a low-dimensional to a high-dimensional space. In higher-dimension, you can solve a linear problem that's a nonlinear in lower-dimensional space.
* Support vector machines are a binary classifier and additional methods can be extended to multi-classification.



## Reference
Harrington, Peter. Machine learning in action. Vol. 5. Greenwich, CT: Manning, 2012.
