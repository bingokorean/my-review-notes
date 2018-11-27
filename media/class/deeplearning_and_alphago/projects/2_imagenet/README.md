# Image (ILSVRC 2012) Classification with pretrained AlexNet

AlexNet is a convolutional neural network that consists of 5 convolutional layers and 3 fully-connected layers.
This neural network was first proposed by Alex Krizhevsky, *et al.* [[1]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), which won the first place at ILSVRC 2012, achieving a top-5 error rate of 15.3%. 

This code represents that how to classify an image using pre-trained AlexNet with ILSVRC 2012 dataset. *ILSVRC* (Imagenet Large Scale Visual Recognition Challenge) is an annual competition for image classification using many (~million) images in 1,000 categories. 

![](./image/cat.jpg)

In ```alexnet_classify.py```...<br>
Since training the whole process with AlexNet requires very large resources and much time, we just load a weight set pre-trained by AlexNet. And We read an image, and perform image classification that outputs 5 most probable categories and their probabilities. 

## Code Brief descriptions
1. Load an image 'cat'
   * we load a test image 'cat.jpg' and resize it to 256x256. If the image is not RGB format, then we convert it to an RGB image.
   * the shape of loaded data is (256, 256, 3)
2. Cropping the loaded image
   * we obtain 10 images of size 227x227 by shifting and cropping the image
   * the shape of cropped and loaded image is (10, 227, 227, 3)
3. Subtract mean to the result of 2
   * subtracting the dataset mean serves to "center" the data
   * we load ```ilsvrc_2012_mean.npy``` which is the mean of ILSVRC Image dataset. It can be downloaded from [here]( https://github.com/BVLC/caffe/blob/master/python/caffe/imagenet/ilsvrc_2012_mean.npy)
   * this is a kind of normalization. It helps for a model to have good local optimums. 
   * since the weights of AlexNet are trained given data which subtracted by mean, we also subtract mean to test data (=cat)
4. Load a weight set pre-trained by AlexNet
   * we load ```bvlc_alexnet.npy``` as a numpy array. It can be downloaded from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
5. Construct the AlexNet
   * among five number of convolutional layers, we don't have maxpooling in two number of convolutional layers.
6. Evaluate the AlexNet
   * we don't need to train the model

   
## Code Details
```
for x in net_data:
    exec ("%s = %s" % (str(x) + "W", "tf.Variable(net_data[x][0])"))
    exec ("%s = %s" % (str(x) + "b", "tf.Variable(net_data[x][1])"))
```
This defines Tesorflow variables for weights and biases for each layer. 'x' will be one of 'conv1'. 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', i.e., names for 5 convolutional layers and 3 fully-connected layers. For the first convolutional layer, the variable 'x' will be 'conv1' and the above two lines containing 'exec' will define "conv1W=tf.Variable(net_data['conv1'][0]" and "conv1b=tf.Variable(net_data['conv1'][1])". The for loop will run 8 times to define Tensorflow variables for weights and biases for all 8 layers.
```
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding=“VALID", group=1)
```
This function constructs a convolutional layer with ‘c_o’ convolutional filters with size (k_h)x(k_w). The stride size (s_h)x(s_w). AlexNet was designed to run on two GPU’s and it has two different paths as explained in lecture notes. ‘group’ indicates the number of groups the convolutional filters are split into. This does not mean you need to run it on a computer with two GPU’s. It can still run even if you have only one GPU provided that there’s enough memory during training.
```
conv1 = tf.nn.relu(conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding=“VALID", group=1))
```
This defines the first convolutional layer which consists of 96 11x11 filters. This takes ‘x’ as an input. Stride sizes are 4x4 and conv1W and conv1b are a weight matrix and a bias vector, respectively, loaded from file "bevlc_alxnet.npy”. “tf.nn.relu” is the ReLU function.
```
lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
```
This normalizes the output of the previous layer. Specifically, this outputs (input/(bias+alpha x sqr_sum)^beta) where ‘sqr_sum’ means squared sum of inputs within depth_radius. For details, refer to [[1]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
```
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
```
This performs max pooling, i.e., taking the maximum value in each 3x3 image patch from the input and moving 2 units horizontally and vertically (stride size is 2x2).
```
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(pylab.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
```
This is a fully connected layer that takes the output of the previous max-pooling layer, reshapes it, multiplies it by a matrix fc6W, and then adds the bias vector fc6b. Finally, this applies the ReLU activation function to the resulting vector.
```
y_conv = tf.nn.softmax(fc8)
```
This computes the softmax values as the output of AlexNet. The output of AlexNet is a 1000x1 vector whose i-th element indicates the estimated likelihood that the input image belongs to the i-th category. We choose 5 most probable categories and print them along with their softmax values. Names of categories are given in ‘caffe_classes.py’.


## Acknowledgement
> EE488C Special Topics in EE Deep Learning and AlphaGo, Fall 2016 & [Information Theory & Machine Learning Lab](http://itml.kaist.ac.kr), School of EE, KAIST & Jongmin Yoon 
