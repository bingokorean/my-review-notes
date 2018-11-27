# Confirm global and local minimum from XOR problem

Our problem is to solve XOR problem using simple neural network where there is only 1 hidden layer. We could try to change the initial point to see whether gradient descent will be stuch at a local minimum, will exhibit slowing down behavior near a saddle point, will oscillate, or will diverge.


The model diagram is as follows:
![](https://github.com/gritmind/review-media/blob/master/class/deeplearning_and_alphago/projects/1_xor_problem/image/nn_for_xor.PNG)

<br>
Training data is as follows: notice that test data is the same as training data

x_data | y_data
-------|--------
[0, 0] | [0]
[0, 1] | [1]
[1, 0] | [1]
[1, 1] | [0]

<br>
The condition for local minimum is as follows:

![](https://github.com/gritmind/review-media/blob/master/class/deeplearning_and_alphago/projects/1_xor_problem/image/ex_for_local.PNG)

* So, as above image, when a = -1, the model is stuck on local minimum. 
* As you can see code, when we in local minimum, the cost is relatively much higher than in global minimum.

<br>
> This resource is based on EE488C Special Topics in EE <Deep Learning and AlphaGo> Fall 2016, School of EE, KAIST
