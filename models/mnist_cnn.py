import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from models.tf_extension import restore_model_params, get_model_params


""" Loading training set """
mnist = input_data.read_data_sets("/tmp/data/")

""" Building CNN model """
height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_size = 3
conv1_stride = 1
conv1_padding = 'SAME'

conv2_fmaps = 64
conv2_size = 3
conv2_stride = 2
conv2_padding = 'SAME'
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps
pool3_dropout_rate = 0.5

n_fc1 = 64 # Fully connected layer
n_outputs = 10

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=(None), name='y')
    training = tf.placeholder_with_default(False, None, name='training')

conv1 = tf.layers.conv2d(X_reshaped, conv1_fmaps, conv1_size, conv1_stride, conv1_padding,
                         activation=tf.nn.relu, name='conv1')
conv2 = tf.layers.conv2d(conv1, conv2_fmaps, conv2_size, conv2_stride, conv2_padding,
                         activation=tf.nn.relu, name='conv2')

with tf.name_scope('pool3'):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)
    # Because of the second layers' stride and pool3 layers' stride both are two,
    # So the shape of pool3 should be 7 * 7,
    # and when connected to fully connected layers, we need to flatten it.

with tf.name_scope('fc1'):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name='fc1')
    fc1_drop = tf.layers.dropout(fc1, pool3_dropout_rate, training=training)

with tf.name_scope('output'):
    logits = tf.layers.dense(fc1, n_outputs, name='output')
    Y_prob = tf.nn.softmax(logits, name='Y_prob')

with tf.name_scope('train'):
    # Apply softmax activation function and then computing its cross entropy.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    # Say whether the targets are in the top K predictions.
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

""" Training parameters"""
# Training with mini-Batch
n_epochs = 100
batch_size = 50

best_loss_val = np.infty
check_interval = 500
check_since_last_progress = 0
max_check_without_progress = 20
best_model_param = None


with tf.Session() as sess:
    init.run()
    # saver.restore(sess, './tf_tmp_checkpoints/mnist_cnn_model')
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: mnist.validation.images,
                                                y: mnist.validation.labels})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    best_model_param = get_model_params()
                else:
                    check_since_last_progress += 1
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_validation = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
        print("Epoch {}, training accuracy: {:.4f}%, validation accuracy: {:.4f}%, best loss: {:.6f}".format(
            epoch, acc_train*100, acc_validation*100, best_loss_val
        ))
        if check_since_last_progress >= max_check_without_progress:
            print("Early stop!")
            break

    if best_model_param:
        restore_model_params(best_model_param)
    acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                        y: mnist.test.labels})
    print("Final accuracy on test set is {:.4f}".format(acc_test*100))


    save_path = saver.save(sess, "./tf_tmp_checkpoints/mnist_cnn_model")


