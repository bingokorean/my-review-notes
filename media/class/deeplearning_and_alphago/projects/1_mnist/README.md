# Solve MNIST problem using CNN and Softmax regression

## MNIST dataset
* database of handwritten digits 0~9
* train set: 55000 / validation set: 5000 / test set: 10000
* each image is gray scale / its dimension is 28 pixels by 28 pixels

## Softmax regression
- Softmax regression or multinomial logistic regression is a generalization of logistic regression to the case where we want to handle multiple classes.
- Softmax performs component-wise exponential function with normalization. Therefore the output of the softmax layer behaves as a probability mass function (i.e., each output is between 0 and 1 and the sum of all outputs is 1) and indicates the likelihood of each output.
- Since the model is simple, its performance does not improve much even if we increase the number of epochs. 

## 2-layer CNN + dropout
### First Convolutional layer
```
x_image = tf.reshape(x, [-1, 28, 28, 1])
```
First we should reshape the input image x to (batch_size)x28x1. -1 means indefinite, which will be automatically calculated to match the total size, e.g., it becomes equal to the number of examples in a minibatch during training and it becomes equal to the number of examples in the test set during evaluation using the test set. The last dimension indicates the number of channels of an image, i.e., a monochrome image has 1 channel and a full color RGB imgae has 3 channels. 
```
W_conv = tf.Variable(tf.truncated_normal([5,5,1, 30], stddev=0.1))
```
This layer consists of 30 convolutional filters and each convolutional filter has size of 5x5x1. The weights of filters are initialized as truncated normmal random variables whose mean is 0 and standard deviation is 0.1.
```
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
```
This is a 30x1 bias vector of the convolutional layer whose elements are initialized as 0.1 constant. 
```
h_conv = tf.nn.conv2d(x_image, W_CONV, strides=[1,1,1,1], padding='VALID')
```
* This means using neural network forms, we apply the 2D convolutional filter (W_conv) to image (x_image).
* 'strides' specify the decimation factors for each dimension of the input, e.g., 2 stride means skipping every other sample. We must fix strides[0]=srides[3]=1, and strides[2] and strides[3] indicate the horizontal and vertical strides, respectively.
* 'padding' means the type of padding algorithm to use. 'VALID' means each output of convolution is from a valid region in the input. If padding='VALID', the output size becomes (28-5+1)x(28-5+1)x30 = 24x24x30. If padding='SAME', the output size is the same as the input by applying zero padding at the input, i.e., 28x28
```
h_relu = tf.nn.relu(h_conv + b_conv)
```
After applying the convolutional filter, we add the bias vector, and then we apply the ReLU activation function, f(x)=max(0,x).
```
h_pool = tf.nn.max_pool(h_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
```
We apply the max pooling to the output of ReLU function. We take the maximum over the 2x2 block of the output, and the strides is 2x2. So, there is no overlap among pooling windows.The output of the pooling layer is 12x12x30, i.e., the sizes become half for horizontal and vertical dimensions. Padding='SAME' or 'VALID' does not matter here because 24 (input size) is divisible by 2 (stride size). If not, then 'VALID' will produce a smaller size than 'SAME' because it will discard the last leftover sample. Note that 'SAME' does not mean the output size is the same as the input size. The output size will be [n/2] if 'VALID' and will be [n/2] if 'SAME'.

### Second Convolutional layer
```
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 30, 50], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[50]))
h_conv2 = tf.nn.conv2d(h_pool, W_conv2, strides=[1, 1, 1, 1], padding='VALID') 
h_relu2 = tf.nn.relu(h_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```
* The size of the convolutional filter is 3x3 for the second convolutional layer
* The size of 'h_conv2' is (batch)x(12-3+1)x(12-3+1)x50 = (batch)x10x10x50.
* The size of 'h_pool2' is (batch)x5x5x50.

### Fully-connected layer
```
W_fc1 = tf.Variable(tf.truncated_normal([5*5*50, 500], stddev=0.1))
```
This layer maps the 5x5x30 feature vector into a 500x1 vector, which is the hidden layer of a fully-connected neural network. The reason why this layer is called the fully-connected layer is because every unit in this layer is connected to every input of this layer. Parameters of the layer are initialized as truncated normal random variables whose mean is 0 and standard deviation is 0.1.
```
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
```
This is a 500x1 bias vector of the fully-connected layer. We initialize it with a constant 0.1.
```
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
```
We reshape the output of the previous pooling layer to (batch_size)x12x12x30. -1 means indefinite.
```
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
```
After multiplying by W_fc1 and adding the bias, we apply the ReLU activation function.

### Dropout layer
Dropout is known to be effective to prevent the neural network from overfitting. This operation randomly drops units from the neural network during training.
* After the first fully-connected layer, we add the following definitions, which applies dropout to the output h_fc1 with keep probability equal to 'keep_prob', whose value will be specified later.
   * ```keep_prob = tf.placeholder(tf.float32)```
   * ```h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)```
* When computing y_hat, we use 'h_fc_drop' instead of 'h_fc1' as shown below.
   * ```y_hat = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2```
* When training and evaluating, 'keep_prob' needs to be specified as follows
   * ```train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})```
   * ```train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.})```
   * ```val_accuracy = accruacy.eval(feed_dict={x:mnist.validation.images, y_:mnist.validation.labels, keep_prob:1.0})```
   * ```test_accuracy = accruacy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})```
* Note that keep_prob is set to 0.5 during training, but it is set to 1.0 for evaluation because we don't want to drop any units during evaluation.

### Output layer (Second fully connected layer)
```
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
```
This layer maps the 500x1 feature vector into 10x1 vector that corresponds to output classes 0~9. Parameters of the layer are initialized as truncated normal random variables whose mean is 0 and standard deviation is 0.1.
```
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
```
This is a 10x1 bias vector of the output layer. We initialize it with a contant 0.1.
```
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
```
Finally, we compute the softmax output values after multiplying h_fc1 by W_fc2 and adding the bias.

#### Addition...
* Zero initialization
   * ```W_conv = tf.Variable(tf.zeros([5,5,1,30]))```
* Uniform initialization
   * ```W_conv = tf.Variable(tf.random_uniform([5,5,1,30])))```
* tanh function
   * ```h_fc1 = tf.tanh(tf.matmul(h_pool_flat, W_fc1)+b_fc1)```


The accuracy of classification gets better as the neural network becomes more complex.

> This resource is based on EE488C Special Topics in EE <Deep Learning and AlphaGo> Fall 2016, School of EE, KAIST
