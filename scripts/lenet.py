import tensorflow as tf
from tensorflow.contrib.layers import flatten

def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID') + b
  return x

def maxpool2d(x, k=2):
  x = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
  return x

def build_lenet(x):
  # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
  mu = 0
  sigma = 0.1
  dropout = 0.75

  weights = {
    'fw1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma)),
    'fw2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma)),
    'fcw1': tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma)),
    'fcw2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma)),
    'fcw3': tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
  }

  biases = {
    'b1': tf.Variable(tf.zeros(6)),
    'b2': tf.Variable(tf.zeros(16)),
    'fcb1': tf.Variable(tf.zeros(120)),
    'fcb2': tf.Variable(tf.zeros(84)),
    'fcb3': tf.Variable(tf.zeros(43))
  }

  # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
  conv1 = conv2d(x, weights['fw1'], biases['b1'])

  # Activation.
  conv1 = tf.nn.relu(conv1)

  # Pooling. Input = 28x28x6. Output = 14x14x6.
  conv1 = maxpool2d(conv1)

  # Layer 2: Convolutional. Output = 10x10x16.
  conv2 = conv2d(conv1, weights['fw2'], biases['b2'])

  # Activation.
  conv2 = tf.nn.relu(conv2)

  # Pooling. Input = 10x10x16. Output = 5x5x16.
  conv2 = maxpool2d(conv2)

  # Flatten. Input = 5x5x16. Output = 400.
  conv2 = flatten(conv2)

  # Layer 3: Fully Connected. Input = 400. Output = 120.
  conv3 = tf.add(tf.matmul(conv2, weights['fcw1']), biases['fcb1'])

  # Activation.
  conv3 = tf.nn.relu(conv3)

  # Layer 4: Fully Connected. Input = 120. Output = 84.
  conv4 = tf.add(tf.matmul(conv3, weights['fcw2']), biases['fcb2'])

  # Activation.
  conv4 = tf.nn.relu(conv4)
  #     conv4 = tf.nn.dropout(conv4, dropout)

  # Layer 5: Fully Connected. Input = 84. Output = 43.
  logits = tf.add(tf.matmul(conv4, weights['fcw3']), biases['fcb3'])

  return logits