import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images


# Load sample images
datasets = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = datasets.shape

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: datasets})



