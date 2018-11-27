# EE488C Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2016, School of EE, KAIST

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Convolutional layer
with tf.name_scope('Conv1'):
    x_image = tf.reshape(x, [-1,28,28,1])
    W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
    b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
    h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
    h_relu = tf.nn.relu(h_conv + b_conv)
    h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
with tf.name_scope('FC'):
    W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
    h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Output layer
with tf.name_scope('Output'):
    W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    y_hat=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
with tf.name_scope('CrossEntropy'):
    cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
    tf.scalar_summary("cost", cross_entropy)

with tf.name_scope('Training'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary("accuracy", accuracy)

summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter("proj1", graph=tf.get_default_graph())
    print("=================================")
    print("|Epoch\tBatch\t|Train\t|Val\t|")
    print("|===============================|")
    avg_cost=0
    batch_count = int(mnist.train.num_examples/100)
    for j in range(5):
        for i in range(batch_count):
            batch = mnist.train.next_batch(100)
            _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1]})
            writer.add_summary(summary, j * batch_count + i)
            if i%50 == 49:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
                val_accuracy = accuracy.eval(feed_dict=\
                    {x: mnist.validation.images, y_:mnist.validation.labels})
                print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
    print("|===============================|")
    test_accuracy = accuracy.eval(feed_dict=\
        {x: mnist.test.images, y_:mnist.test.labels})
    print("test accuracy=%.4f"%(test_accuracy))

