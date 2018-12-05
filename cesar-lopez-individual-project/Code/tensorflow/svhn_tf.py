"""
Group 1: Cesar Lopez, Akshay Akmath, Purvi Thakor

Convolutional Models for The Street View House Numbers (SVHN) Dataset

Machine Learning 2

This file runs a CNN model using the tensorflow framework.
At the end of the code, there is a code you will need to copy and paste into your terminal.
This will output the a link you can copy and paste on your browser. This will open tensorboard.

"""
# Importing libraries
from svhnpck.data_loader import SvhnData
from svhnpck.svhn_formatter import change_range
import warnings
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.client import device_lib

warnings.filterwarnings("ignore")  # Trying to filter out warnings
print(device_lib.list_local_devices())  # print out all devices in your working machine.
real_start = time.time()  # to track how much total time the total training takes. This is will be the starting time.

# ----------------------------------------Load the Data-------------------------------------------
'''
I created a package (svhnpck) and in that package I created the class (SvhnData).
This class collect the data from the bucket.
If you call the get_data() methods, it will return x_train, y_train, x_test, y_test.
'''
svhn = SvhnData()  # create an object of the class
svhn.load_data()  # load the data from the bucket
x_train, y_train, x_test, y_test = svhn.get_data(change_dim=True, one_hot=True)  # get the train and test data

x_train, y_train = change_range(300000, x_train, y_train)
x_test, y_test = change_range(40000, x_test, y_test)

# -------------------------------------Creating Place Holders-------------------------------------

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])  # graph that will take training images.
y_true = tf.placeholder(tf.float32, shape=[None, 10])    # graph that will take training labels.
# hold_prob = tf.placeholder(tf.float32)  # no dim needed will be used as probability hold for dropout nodes

# ---------------------------------------Creating Layers-------------------------------------------

'''

Originally, I wrote this code in a modular approach. I wrote a function for:

  -initializing Variables.
  -Convo layers
  -Max pooling
  -fully connected layers.

This; however, caused issue with creating the graph and histograms for tensorboard.
Therefore, I had to rewrite the code in a way that I had full control in customizing the tensorboard
the way I wanted...I still have much to learn.

'''
# scope for the first convolution layer (tensorboard)
with tf.name_scope('convo_layer1') as scope:
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev=0.1), name='weight1')
    b1 = tf.Variable(tf.constant(0.1, shape=[16]), name='bias1')
    convolution_1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME', name='convo1')
    convo_layer1 = tf.nn.relu(convolution_1 + b1)
    W1_hist = tf.summary.histogram("weights1", W1)  # create tensorboard histogram for filter
    b1_hist = tf.summary.histogram("bias1", b1)  # create tensorboard histogram for bias
    convo_layer1_hist = tf.summary.histogram("convo_layer1", convo_layer1)

# scope for the first max pooling layer (tensorboard)
with tf.name_scope('max_layer1') as scope:
    max_layer1 = tf.nn.max_pool(convo_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max_1_hist = tf.summary.histogram('max_layer1', max_layer1)

# scope for the second convolution layer (tensorboard)
with tf.name_scope('convo_layer2') as scope:
    W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name='weight2')
    b2 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias2')
    convolution_2 = tf.nn.conv2d(max_layer1, W2, strides=[1, 1, 1, 1], padding='SAME', name='convo2')
    convo_layer2 = tf.nn.relu(convolution_2 + b2)
    W2_hist = tf.summary.histogram("weights2", W2)  # create tensorboard histogram for filter
    b2_hist = tf.summary.histogram("bias2", b2)  # create tensorboard histogram for bias
    layer2_hist = tf.summary.histogram("convo_layer2", convo_layer2)

# scope for the second max pooling layer (tensorboard)
with tf.name_scope('max_layer2') as scope:
    max_layer2 = tf.nn.max_pool(convo_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    max_2_hist = tf.summary.histogram('max_layer2', max_layer2)

# Scope for the two fully connected layers and one drop out node in between (tensorboard)
with tf.name_scope('connected_layer_1') as scope:
    flat_max2_output = tf.reshape(max_layer2, [-1, 8*8*32])
    input_size_max2 = int(flat_max2_output.get_shape()[1])

    W3 = tf.Variable(tf.truncated_normal([input_size_max2, 500], stddev=0.1), name='weight3')
    b3 = tf.Variable(tf.constant(0.1, shape=[500]), name='bias3')

    vect_convert = tf.matmul(flat_max2_output, W3) + b3
    full_connected_one = tf.nn.relu(vect_convert, name='full_connected_one')
    full_one_dropout = tf.nn.dropout(full_connected_one, keep_prob=0.2)
    W3_hist = tf.summary.histogram("weights3", W3)  # create tensorboard histogram for filter
    b3_hist = tf.summary.histogram("bias3", b3)  # create tensorboard histogram for bias
    fc1_hist = tf.summary.histogram("full_connected_one", full_connected_one)

with tf.name_scope('connected_layer_2') as scope:
    input_size_connected = int(full_one_dropout.get_shape()[1])
    W4 = tf.Variable(tf.truncated_normal([input_size_connected, 10], stddev=0.1), name='weight4')
    b4 = tf.Variable(tf.constant(0.1, shape=[10]), name='bias4')
    y_pred = tf.matmul(full_one_dropout, W4) + b4
    W4_hist = tf.summary.histogram("weights4", W4)  # create tensorboard histogram for filter
    b4_hist = tf.summary.histogram("bias4", b4)  # create tensorboard histogram for bias
# ---------------------------------------Loss Function---------------------------------------------

# This has softmax which is the output layer.
# But then uses cross_entropy as the loss function
cross = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
cross_entropy = tf.reduce_mean(cross)
cost_scalar = tf.summary.scalar("Training Loss", cross_entropy)  # creating a scalar viz for tensorboard

# ---------------------------------------Optimizer-------------------------------------------------

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

# -------------------------------------Getting the Accuracy----------------------------------------
matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(matches, tf.float32))
acc_scalar = tf.summary.scalar('Testing accuracy', acc)


# mistake = tf.not_equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
# error = tf.reduce_mean(tf.cast(mistake, tf.float32))
# error_scalar = tf.summary.scalar('Testing Loss', error)
# -------------------------------Initializing tensorboard writer-----------------------------------

TB_writer = tf.summary.FileWriter("./my_log_dir")  # The writer for the histograms to be in tensorboard
summaries = tf.summary.merge_all()  # Merging all the summaries created for them to be able to be displayed the vis

# -----------------------------------Batch and Epoch Sizes-----------------------------------------
batch_size = 500
epochSize = 10

print('Training Data length: ' + str(x_train.shape[0]))
num_of_batches = int(x_train.shape[0]/batch_size)
print('Number of Batches: ' + str(num_of_batches))
print('\n')

# ---------------------------------------The Session-----------------------------------------------

saver = tf.train.Saver()  # Will save a the model
iter_time = np.zeros(epochSize)  # To save time it takes for each iteration (epochs)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_log_dir', sess.graph)  # The writer for the graph to be in tensorboard
    for k in range(epochSize):
        temp_var = 0
        start_time = time.time()
        for i in range(num_of_batches):
            x_batch = x_train[temp_var:temp_var + batch_size]  # Getting the next batches.
            y_batch = y_train[temp_var:temp_var + batch_size]  # Getting the next batches.
            temp_var += batch_size
            # Runs the operations of the graph
            sess.run(train, feed_dict={X: x_batch, y_true: y_batch})

        # Get the training error. If the pred is not correct
        print('\n\nCurrently on epoch {}'.format(k+1))

        print('Test Accuracy is: ', end='')
        # Test Accuracy. In the testing, it checks if pred is correct
        print(sess.run(acc, feed_dict={X: x_test, y_true: y_test}))

        # print('Test Loss is: ', end='')
        # print(sess.run(error, feed_dict={X: x_test, y_true: y_test}))

        endtime = time.time() - start_time  # get the total time is current epoch
        print('Runtime: ' + str(round(endtime, 2)) + ' Seconds\n')
        iter_time[k] = round(endtime, 2)

        # Runs the summaries to be added to the tensorboard
        summ = sess.run(summaries, feed_dict={X: x_batch, y_true: y_batch})
        TB_writer.add_summary(summ, global_step=k)

        writer.close()
        # Saves the model
        # save_path = saver.save(sess, "./tf_models/mymodel.ckpt")
        # print("Model saved in path: %s" % save_path)
# gets the time it took for the whole training to complete
Rendtime = time.time() - real_start
print('Full Runtime: ' + str(round(Rendtime, 2)) + ' Seconds\n\n')
# Prints out the average runtime per epoch.
print('average runtime per epoch is ' + str(round(np.mean(iter_time), 2)) + ' seconds')

# Type this on Terminal in Same Directory as the tensorflow code
# tensorboard --logdir ./my_log_dir/

# END
