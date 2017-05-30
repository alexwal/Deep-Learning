# Author: Alex Walczak, May 2017
# Created with Python 2.7, TensorFlow r1.1
from __future__ import division
from __future__ import print_function 
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
from six.moves import xrange # pylint: disable=redefined-builtin

sqrt = math.sqrt

'''
  # Train TF model on small cifar10 dataset.
  # Overview of functions available:

  # Compute input images and labels for training, assigning appropriate placeholders.
  feed_dict = fill_feed_dict(<placeholders>)

  # Compute inference on the model inputs to make a prediction.
  predictions = inference(inputs, phase_placeholder)

  # Compute the total loss of the prediction with respect to the labels.
  loss = loss(predictions, labels)

  # Create a graph to run one step of training with respect to the loss.
  train_op = train(loss, global_step)
  
  # Notes:
  images shape: (10000, 32, 32, 3)
  labels shape: (10000,)

'''

# Helper classes for obtaining batches from data and creating datasets

class BatchProvider(object): # aka a single 'data_set' (train or test)
  def __init__(self, data, labels, is_train, shuffle=True):
    assert len(data) == len(labels), 'Every data point must have a corresponding label (first dimensions must match).'
    self.data = data
    self.labels = labels
    self.idx = 0
    self.is_train = is_train
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
      # print('Reached end of data. Extending this batch by wrapping around to front.')
      self.idx = self.idx % len(self.data)
      additional_data = self.data[:self.idx]
      additional_labels = self.labels[:self.idx]
      batch = np.concatenate((batch, additional_data))
      batch_labels = np.concatenate((batch_labels, additional_labels))
    return batch, batch_labels

  def __len__(self):
    return len(self.data)

class Read_data_sets(object):
  def __init__(self, data_dir, train_data_file, train_labels_file,
                test_data_file, test_labels_file):
    # Load data files
    train_data = np.load(os.path.join(data_dir, train_data_file)) 
    train_labels = np.load(os.path.join(data_dir, train_labels_file)) - 1 #TODO: get rid of -1 <<-- --vv
    test_data = np.load(os.path.join(data_dir, test_data_file)) 
    test_labels = np.load(os.path.join(data_dir, test_labels_file)) - 1
    # Create batch providers
    self.train = BatchProvider(train_data, train_labels, is_train=True) # usage ex: Read_data_sets.train.next_batch(batch_size)
    self.test = BatchProvider(test_data, test_labels, is_train=False)

# Read test and train data set files
data_sets = Read_data_sets('small-cifar10-data/', 
                           'cifar10-train-data.npy', 'cifar10-train-labels.npy',
                           'cifar10-test-data.npy', 'cifar10-test-labels.npy')

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
im_H, im_W = 32, 32
batch_size = 16
default_learning_rate = 1e-2
default_dtype = tf.float32

# UTILS:

def spatial_batch_norm(inputs, train_phase, decay=0.999, name="SBN", epsilon=1e-3):
    # train_phase: a tf.placeholder(tf.bool) that switches between training/testing (ie using batch or pop mean/var)
    channels = inputs.get_shape()[-1]
    scale = tf.Variable(tf.ones([channels]))
    offset = tf.Variable(tf.zeros([channels]))
    pop_mean = tf.Variable(tf.zeros([channels]), trainable=False)
    pop_var = tf.Variable(tf.ones([channels]), trainable=False)
    def if_train():
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2]) # compute mean across these axes (all but channels)
        # Exponential Mov. Avg. Decay (compute moving average of population, update as batches are seen.)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var  = tf.assign(pop_var,  pop_var  * decay + batch_var  * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]): # makes sure the moving averages are updated during training (absent below:)
          return tf.identity(batch_mean), tf.identity(batch_var)
    def if_not_train():
        return tf.identity(pop_mean), tf.identity(pop_var)
    mean, var = tf.cond(train_phase, if_train, if_not_train)
    return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name)

def make_var(name, shape, stddev=None, constant=None):
  # On CPU. Later: add weight regularization
  assert (stddev is None) != (constant is None), 'Stddev and constant cannot both be None: one determines the initializer used.'
  with tf.device('/cpu:0'):
    if stddev is not None:
      initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=default_dtype) # values stay w/in 2 stddev of mean (default mean = 0)
    elif constant is not None:
      initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=default_dtype)
  return var

def fill_feed_dict(data_set, images_pl, labels_pl, train_phase_pl, keep_prob_pl, keep_prob):
  batch = data_set.next_batch(batch_size)
  return {images_pl: batch[0], labels_pl: batch[1], train_phase_pl: data_set.is_train, keep_prob_pl: keep_prob}

def placeholder_inputs():
  images_pl = tf.placeholder(default_dtype, shape=[None, im_H, im_W, 3])
  labels_pl = tf.placeholder(tf.int32, shape=[None]) 
  train_phase_pl = tf.placeholder(tf.bool, name='train_phase')
  keep_prob_pl = tf.placeholder(tf.float32)
  return images_pl, labels_pl, train_phase_pl, keep_prob_pl

# STAGES:

def inputs(data_set):
  '''Compute input images and labels for training/testing.'''
  image_batch, label_batch = data_set.next_batch(batch_size)
  return image_batch, label_batch

def inference(images, train_phase, keep_prob):
  '''Compute inference on the model inputs to make a prediction.'''

  '''
    LeNet:
    model:add(nn.SpatialConvolution(3, 6, 5, 5)) 
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.SpatialConvolution(6, 16, 5, 5))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.View(16*5*5))
    model:add(nn.Linear(16*5*5, 120))
    model:add(nn.ReLU())
    model:add(nn.Linear(120, 84))
    model:add(nn.ReLU())
    model:add(nn.Linear(84, 10))
    model:add(nn.LogSoftMax())
  '''

  # conv1 [32 x 32 x 3 -> ] 
  with tf.variable_scope('conv1') as scope:
    kernel = make_var('weights', [5, 5, 3, 6], stddev=sqrt(1.0 / (5 * 5 * 3))) # Initialize weights by drawing from distribution w/ stddev=1/sqrt(fan-in)
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='VALID') # For VALID conv: output dims: int((inW - kW)/stride + 1)
    biases = make_var('biases', [6], constant=0.05)
    pre_activation = tf.nn.bias_add(conv, biases)
    post_activation = tf.nn.relu(pre_activation)
    conv1 = spatial_batch_norm(post_activation, train_phase, name=scope.name)

  # pool1 (2x2) 
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1')
  dropout1 = tf.nn.dropout(pool1, keep_prob)

  # conv2 
  with tf.variable_scope('conv2') as scope:
    kernel = make_var('weights', [5, 5, 6, 16], stddev=sqrt(1.0 / (5 * 5 * 6)))
    conv = tf.nn.conv2d(dropout1, kernel, strides=[1, 1, 1, 1], padding='VALID')
    biases = make_var('biases', [16], constant=0.1)
    pre_activation = tf.nn.bias_add(conv, biases)
    post_activation = tf.nn.relu(pre_activation)
    conv2 = spatial_batch_norm(post_activation, train_phase, name=scope.name)


  # pool2 
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool2')
  dropout2 = tf.nn.dropout(pool2, keep_prob)

  # fc1 
  with tf.variable_scope('fc1') as scope:
    flattened = tf.reshape(dropout2, [-1, 5 * 5 * 16])
    W = make_var('weights', [5 * 5 * 16, 120], stddev=sqrt(1.0 / (5 * 5 * 16)))
    linear = tf.matmul(flattened, W)
    biases = make_var('biases', [120], constant=0.1)
    pre_activation = tf.add(linear, biases)
    post_activation = tf.nn.relu(pre_activation, name=scope.name)
    fc1 = tf.nn.dropout(post_activation, keep_prob)

  # fc2 
  with tf.variable_scope('fc2') as scope:
    W = make_var('weights', [120, 84], stddev=sqrt(1.0 / 120)) 
    linear = tf.matmul(fc1, W)
    biases = make_var('biases', [84], constant=0.1)
    pre_activation = tf.add(linear, biases)
    post_activation = tf.nn.relu(pre_activation, name=scope.name)
    fc2 = tf.nn.dropout(post_activation, keep_prob)

  # logits [-> num_classes=10]
  with tf.variable_scope('logits') as scope:
    num_classes = len(classes)
    W = make_var('weights', [84, num_classes], stddev=sqrt(1.0 / 84))
    linear = tf.matmul(fc2, W)
    biases = make_var('biases', [num_classes], constant=0.1)
    pre_activation = tf.add(linear, biases)
    logits = tf.nn.relu(pre_activation, name=scope.name)

  return logits

def loss(logits, labels):
  labels = tf.cast(labels, tf.int32)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
  return tf.reduce_mean(cross_entropy, name='mean_xent')

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

def do_eval(sess, eval_correct, images_pl, labels_pl, train_phase_pl, data_set, keep_prob_pl, keep_prob=1.0, max_batches=None):
  '''Evaluation over one epoch (all samples) in data_set.'''
  num_batches = len(data_set) // batch_size
  if max_batches is not None:
    num_batches = min(num_batches, max_batches)
  num_samples = num_batches * batch_size
  correct_count = 0
  for batch in xrange(num_batches):
    feed_dict = fill_feed_dict(data_set, images_pl, labels_pl, train_phase_pl, keep_prob_pl, keep_prob)
    correct_count += sess.run(eval_correct, feed_dict=feed_dict) # sess.run(WHAT_YOU_WANT_TO_KNOW, WHAT_YOU_NEED_TO_PROVIDE)
  accuracy = float(correct_count) / num_samples
  print('Num samples: %d, Num correct: %d, Accuracy: %0.04f' % (num_samples, correct_count, accuracy))

def run_training():
  # Build graph
  images_pl, labels_pl, train_phase_pl, keep_prob_pl = placeholder_inputs()
  logits = inference(images_pl, train_phase_pl, keep_prob_pl)
  total_loss = loss(logits, labels_pl)
  train_op = training(total_loss, default_learning_rate)
  eval_correct = evaluation(logits, labels_pl)
  
  # Setup session
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  # Perform training and evaluation
  for step in xrange(5000): # each step trains on a single batch
    feed_dict = fill_feed_dict(data_sets.train, images_pl, labels_pl, train_phase_pl, keep_prob_pl, keep_prob=0.5)
    _, train_loss_value = sess.run([train_op, total_loss], feed_dict=feed_dict) # discard train_op run output

    if step % 200 == 0:
      print("\nStep %d, step train_loss_value: %0.06f" % (step, train_loss_value))
      print("==> Evaluating training data:")
      do_eval(sess, eval_correct, images_pl, labels_pl, train_phase_pl, data_sets.train, keep_prob_pl, max_batches=10)
      print("==> Evaluating testing data:")
      do_eval(sess, eval_correct, images_pl, labels_pl, train_phase_pl, data_sets.test, keep_prob_pl, max_batches=10)

  print("\nFinal results:")
  print("==> Evaluating training data:")
  do_eval(sess, eval_correct, images_pl, labels_pl, train_phase_pl, data_sets.train, keep_prob_pl)
  print("==> Evaluating testing data:")
  do_eval(sess, eval_correct, images_pl, labels_pl, train_phase_pl, data_sets.test, keep_prob_pl)

if __name__ == '__main__':
  run_training()

