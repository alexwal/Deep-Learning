# Author: Alex Walczak, May 2017
# Created with Python 2.7, TensorFlow r1.1
from __future__ import division
from __future__ import print_function 
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

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

def inputs(data_set):
  image_batch, label_batch = data_set.next_batch(batch_size)
  return image_batch, label_batch

def inference(inputs):
  pass

def loss(predictions, labels):
  pass

def train(loss, global_step):
  pass

