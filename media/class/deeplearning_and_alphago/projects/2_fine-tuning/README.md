# Fine-tuning AlexNet using Caltech 101 dataset

We will re-train the last fully-connected layer of the pre-trained AlexNetto classify new categories of images from Caltech 101 dataset. Caltech 101 dataset contains a set of images in 101 categories such as airplanes, faces, and motorbikes, collected in 2003 by Fei-FeiLi, et al. There are about 40 to 800 images per category. Since the numbers of images vary a lot, we choose 10 categories with about 100 images in each category.The chosen 10 categories and associated image file names can be found in “101_labels_ten.txt”.

<br>
There are materials we need to fine-tuning.

File Name | Description
----------|------------
101_labels_ten.txt | image file names for 10 categories
crop_batch.py | cropping and data augmentation
divide_train_val.py | divide images into train and validation sets
Generate_HDF5.py | generate HDF5 files
ilsvrc_2012_mean.npy | mean information for ILSVRC 2012 dataset
project2_alexnet_finetune.py | code for fine-tuning
project2_HDF5_generation.py | code for generating *.h5 files

We can download these files: <br>
1.```ilsvrc_2012_mean.npy``` which is the mean of ILSVRC Image dataset. It can be downloaded from [here](https://github.com/BVLC/caffe/blob/master/python/caffe/imagenet/ilsvrc_2012_mean.npy)
<br>
2.```bvlc_alexnet.npy``` as a numpy array. It can be downloaded from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
<br>
3. "Caltech 101 dataset" can be downloaded as follows:
```
$ wget--no-check-certificate https://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
$ tar xvzf101_ObjectCategories.tar.gz
```

## Program Process
1. Run ```HDF5_generation.ipynb``` to generate .h5 files in HDF format
   * Hierarchical data format (HDF) is a data format for organizing and storing a large amount of data.
   * ```HDF5_generation.ipynb``` reads ```101_labels_ten.txt``` and calls ```divide_train_val()``` function in ```divide_train_val.py``` to randomly split the images specified in the file ```101_labels_ten.txt``` into a training set and a validation set. The split is 80:20, i.e., 80% for training set and 20% for validation set.
   * After running the code, you will get ```Caltech101_ten_train.h5```, which contains all the training images and their labels and ```Caltech101_ten_val.h5```, which contains all the validation images and their labels.
    
2. Run ```alexNet_finetune.ipynb``` which trains the last layer of AlexNetusing the new dataset.
   * You can see the test accuracy is quite high, which is because there are only 10 categories and also because AlexNetwas already trained using 1,000 categories and you are only re-training the last layer.

## Code Details
```
fan1 = math.sqrt(6.0 / (4096.0 + 10.0))
fc8W = tf.Variable(tf.random_uniform([4096, 10], minval=-fan1, maxval=fan1))
```
We use Xavier initialization
```
train_step = opt.minimize(cross_entropy, var_list=[fc8W,fc8b])
```
We train the last fully-connected layer while fixing other layers. The variables that need to be optimized, i.e., the weight and bias of the last fully connected layer, are specified as ‘var_list’.
```
npy_save = {}
npy_save[0] = fc8W.eval()
npy_save[1] = fc8b.eval()
np.save(output_weight_filename, npy_save)
```
We save the parameters of the last fully-connected layer as a numpy file.

Other explanation is in [here](https://github.com/gritmind/review-media/tree/master/class/deeplearning_and_alphago/projects/2_imagenet)





## Additional work
We use different optimizers, compare them and the selected optimizers are as follows:
   * Gradient descent
   * Momentum 
   * Adagrad
   * Adam

There are two kinds of experiment. First, I compare results on basis of methods. Second, I compare the results on basis of parameters of each method. The modified code is only optimizer function part.
The detail story is in [my report](https://1drv.ms/w/s!AllPqyV9kKUrgX92dNlz7PXWKKjk)


## Acknowledgement
> EE488C Special Topics in EE Deep Learning and AlphaGo, Fall 2016 & Information Theory & Machine Learning Lab, School of EE, KAIST & Jongmin Yoon



