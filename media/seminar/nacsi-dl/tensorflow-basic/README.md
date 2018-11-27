# Tensorflow
   * It is based on Python, which enable to interwork other python packages
   * Symbolic implementation 
      * fast and easy to implement
      * can do differertiation so, we don't need implement 'backpropagation'
   * Same code for both CPU and GPU
   * Computational Graph Abstraction
   * compiled faster than Theano
   * TensorBoard Visualization
   * parallel data and models 
   
### Basic Usage
   * Represents computations as graphs
   * Executes graphs in the context of sessions
   * Represents data as tensors
   * Maintains state with variables
   * Uses feeds and fetches to get data into and out of arbitrary operations

### Session and Graph
1. Computational Graph
   * Tensorflow programs are sturcted into a construction phase, which assembles a graph, and an execution phase that uses a session to execute operations in the graph.

2. Launching the graph in a session
   * After we build a construction phase, we launch a graph by creating a session object. (Construction -> Launching)

<br>
### Variable type of tensorflow diagram

![](https://docs.google.com/drawings/d/1qaFys5F7_FqI6FvVpQDGIDl-XJwLGh0x3SoL4BTAbw0/pub?w=480&amp;h=360)
* Placeholder() is regarded as symbolic variables whose values are not specified at the first time.
* Operator() can be defined as from simple operators such as tf.add(), tf.mul() to complex operators such as tf.reduce_mean(), tf.train.AdagradOptimizer()  
* Operator() consists of Variable(), Constant() and Operator()
* Session.run() can has two kinds of operator(). For example, both tf.reduce_mean() and tf.train.AdagradOptimizer() in RNN  
* Constant() is defined as tf.constant() or any numerical values. 

```
x = tf.placeholder(tf.float32, shape=[None, 784]
y = tf.placeholder(tf.float32, shape=[None, 10]
```
* In Tensorflow, symbolic manipulations are done to calculate gradient automatically. x and y are symbolic variables whose values are not specified yet. They are place holders for 32-bit floating matrices of sizes None*784 and None*10, respectively. 'None' means unspeficied. It will be specified later when the actual data is fed to x and y when you run sess.run() via feed_dict={x: x_data, y: y_data}. 
```
W = tf.Variable(tf.truncated_normal([dim1, dim2], stddev=0.1))
```
* In Tensorflow, a Variable is a modifiable tensor and neural network parameters (weights and biases) are defined as Variables. Variables are usually initialized as a small random amount. Truncation is done so that the absolute value does not exceed 2 times the standard deviation. If we don't specify the mean and standard deviation of the truncated normal function, default values are 0 and 1, respectively.
```
sess = tf.Session()
```
* A session object encapsulates the environment in which operation objects are executed and tensor objects are evaluated. We define a new session 'sess' ahead of the training procedure.


<br>
### Difference between tf.placeholder and tf.Variable

tf.Variable | tf.placeholder
------------|---------------
to be initialized | to do not be initialized
weights/biases | train/test dataset
is modified internally via session | is changed externally via feed_dic={}
is modified by a predefined operators | is changed by a user preference
like global variable | like local variable

* tf.Variable: have to define an initial value when declaring it
* tf.placeholder: don't have to define an initial value and should be specified at run time with feed_dict argument inside Session.run
* Both tf.Variable and tf.placeholder can be regarded as a buffer because they all has a capsule such as contant, feed_dict={tf.placeholder: X_train} 
* tf.Variable seems like a global variable because when after compelting run.session it will be renewed like the global variable.
* tf.Variable is modified by a predefined operators such as gradientdescent_optimizirs which get the operators involving tf.Variables. Instead, tf.placeholder is changed by a user preference logic via feed_dict every session. 
* In addition, before using sess.run() for tf.Variable(), we must execute a sess.run(tf.initialize_all_variables()). Just like to notify a setting of variables to session. 

<br>
> This resource is based on EE488C Special Topics in EE <Deep Learning and AlphaGo> Fall 2016, School of EE, KAIST
> [Biointelligence Laboratory](http://bi.snu.ac.kr) & Department of Computer Science and Engineering & Seoul National University & Eun-Sol Kim

