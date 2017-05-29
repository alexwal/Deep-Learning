# Author: Alex Walczak, May 2017
# Created with Python 2.7, TensorFlow r1.1
from __future__ import division
from __future__ import print_function 
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from six.moves import xrange # pylint: disable=redefined-builtin

'''
  # Train TF model on small cifar10 dataset.
  # Overview of functions available:

  # Compute input images and labels for training. If you would like to run
  # evaluations, use inputs() instead.
  inputs, labels = distorted_inputs()

  # Compute inference on the model inputs to make a prediction.
  predictions = inference(inputs)

  # Compute the total loss of the prediction with respect to the labels.
  loss = loss(predictions, labels)

  # Create a graph to run one step of training with respect to the loss.
  train_op = train(loss, global_step)
  
  # Notes:
  images shape: (10000, 32, 32, 3)
  labels shape: (10000,)

'''

# Helper classes for obtaining batches from data and creating datasets

class BatchProvider(object):
  def __init__(self, data, labels, shuffle=True):
    assert len(data) == len(labels), 'Every data point must have a corresponding label (first dimensions must match).'
    self.data = data
    self.labels = labels
    self.idx = 0
    if shuffle:
      indices = np.random.permutation(len(data))
      self.data = self.data[indices]
      self.labels = self.labels[indices]

  def next_batch(self, batch_size):
    idx = self.idx
    batch = self.data[idx:idx + batch_size]
    batch_labels = self.labels[idx:idx + batch_size]
    self.idx += batch_size
    if len(batch) < batch_size: # self.idx is past end of data.
      # maybe re-shuffle data now?
      print('Reached end of data. Extending this batch by wrapping around to front.')
      self.idx = self.idx % len(self.data)
      additional_data = self.data[:self.idx]
      additional_labels = self.labels[:self.idx]
      batch = np.concatenate((batch, additional_data))
      batch_labels = np.concatenate((batch_labels, additional_labels))
    return batch, batch_labels

class Read_data_sets(object):
  def __init__(self, data_dir, train_data_file, train_labels_file,
                test_data_file, test_labels_file):
    # Load data files
    train_data = np.load(os.path.join(data_dir, train_data_file)) 
    train_labels = np.load(os.path.join(data_dir, train_labels_file)) 
    test_data = np.load(os.path.join(data_dir, test_data_file)) 
    test_labels = np.load(os.path.join(data_dir, test_labels_file)) 
    # Create batch providers
    self.train = BatchProvider(train_data, train_labels) # usage ex: Read_data_sets.train.next_batch(batch_size)
    self.test = BatchProvider(test_data, test_labels)

# Read test and train data set files
data_sets = Read_data_sets('small-cifar10-data/', 
                           'cifar10-train-data.npy', 'cifar10-train-labels.npy',
                           'cifar10-test-data.npy', 'cifar10-test-labels.npy')

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
batch_size = 16
default_dtype = tf.float32
default_learning_rate = 1e-2
im_H, im_W = 32, 32

# UTILS:

def make_variable(name, shape, stddev=None, constant=None):
  # On CPU. Later: add weight regularization
  assert stddev is not None and constant is not None, 'Stddev and constant cannot both be None: one determines the initializer used.'
  with tf.device('/cpu:0'):
    if stddev is not None:
      initializer = tf.truncated_normal(stddev=stddev, dtype=default_dtype) # values stay within two stddevs of mean (default mean = 0)
    elif constant is not None:
      initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=default_dtype)
  return var

def fill_feed_dict(data_set, images_pl, labels_pl):
  batch = data_set.next_batch(batch_size)
  return {images_pl: batch[0], labels_pl: batch[1]}

def placeholder_inputs():
  images_pl = tf.placeholder(default_dtype, shape=[None, im_H, im_W, 3])
  labels_pl = tf.placeholder(tf.int32, shape=[None]) 
  return images_pl, labels_pl

# STAGES:

def inputs(data_set):
  '''Compute input images and labels for training/testing.'''
  image_batch, label_batch = data_set.next_batch(batch_size)
  return image_batch, label_batch

def inference(images):
  '''Compute inference on the model inputs to make a prediction.'''

  # conv1 [32 x 32 x 3 -> 32 x 32 x 64]
  with tf.variable_scope('conv1') as scope:
    kernel = make_var('weights', [5, 5, 3, 64], stddev=0.005)
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = make_var('biases', [64], constant=0.0)
    pre_activation = tf.nn.bias_add(conv, biases)
    # Spatial Batch Norm here?
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1 (2x2) [32 x 32 x 64 -> 16 x 16 x 64]
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # conv2 [16 x 16 x 64 -> 16 x 16 x 48]
  with tf.variable_scope('conv2') as scope:
    kernel = make_var('weights', [3, 3, 64, 48], stddev=0.005)
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = make_var('biases', [48], constant=0.0)
    pre_activation = tf.nn.bias_add(conv, biases)
    # Spatial Batch Norm here?
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # conv3 [16 x 16 x 48 -> 16 x 16 x 24]
  with tf.variable_scope('conv3') as scope:
    kernel = make_var('weights', [3, 3, 48, 24], stddev=0.005)
    conv = tf.nn.conv2d(conv2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = make_var('biases', [48], constant=0.0)
    pre_activation = tf.nn.bias_add(conv, biases)
    # Spatial Batch Norm here?
    conv3 = tf.nn.relu(pre_activation, name=scope.name)

  # pool2 [16 x 16 x 24 -> 8 x 8 x 24]
  pool2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  # fc1 [8 x 8 x 24 -> 8*8*64 -> 128]
  flattened = tf.reshape(pool2, [-1, 8 * 8 * 24])
  with tf.variable_scope('fc1') as scope:
    # Dropout/keep_prob?
    W = make_var('weights', [8 * 8 * 24, 128], stddev=0.001) 
    linear = tf.nn.matmul(pool2, W)
    biases = make_var('biases', [128], const=0.0)
    pre_activation = tf.add(linear, biases)
    fc1 = tf.nn.relu(pre_activation, name=scope.name)

  # logits [128 -> num_classes=10]
  with tf.variable_scope('logits') as scope:
    num_classes = len(classes)
    W = make_var('weights', [128, num_classes], stddev=0.004)
    linear = tf.matmul(fc1, W)
    biases = make_var('biases', [num_classes], constant=0.0)
    pre_activation = tf.add(linear, biases)
    logits = tf.nn.relu(pre_activation, name=scope.name)

  return logits

def loss(logits, labels):
  labels = tf.cast(labels, tf.int32)
  cross_entropy = tf.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
  return tf.reduce_mean(cross_entropy, 'mean_xent')

def training(total_loss, learning_rate):
  # Note: this isn't called over and over again when training; it's called
  # once to build this part of the computational graph; the operations created
  # here are repeated, which themselves increment global_step, update gradients, etc.
  # (Only train_op which performs a minimizing update is repeated.)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(total_loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, k=1)
  # return number true entries
  return tf.reduce_sum(tf.cast(correct, tf.int32))

###

def do_eval(sess, eval_correct, images_pl, labels_pl, data_set):
  '''Evaluation over one epoch (all samples) in data_set.'''
  num_batches = len(data_set) // batch_size
  num_samples = num_batches * batch_size
  correct_count = 0
  for batch in xrange(num_batches):
    feed_dict = fill_feed_dict(data_set, images_pl, labels_pl)
    correct_count += sess.run(eval_correct, feed_dict=feed_dict) # sess.run(WHAT_YOU_WANT_TO_KNOW, WHAT_YOU_NEED_TO_PROVIDE)
  accuracy = float(correct_count) / num_samples
  print('Num samples: %d, Num correct: %d, Accuracy: %0.04f' % (num_samples, correct_count, accuracy))

def run_training():
  # Build graph
  images_pl, labels_pl = placeholder_inputs()
  logits = inference(images_pl)
  total_loss = loss(logits, labels_pl)
  train_op = training(total_loss, default_learning_rate)
  eval_correct = evaluation(logits, labels_pl)
  
  init = tf.global_variables_initializer()
  sess.run(init)

  for step in xrange(1000):
    feed_dict = fill_feed_dict(data_sets.train, images_pl, labels_pl)
    _, loss_value = sess.run([train_op, total_loss], feed_dict=feed_dict) # discard train_op run output

    if step % 100 == 0:
      print("Step %d, step loss_value: %0.04f" % (step, loss_value))
      print("Evaluating training data:")
      do_eval(sess, eval_correct, images_pl, labels_pl, data_set.train)
      print("Evaluating testing data:")
      do_eval(sess, eval_correct, images_pl, labels_pl, data_set.test)

if __name__ == '__main__':
  run_training()




