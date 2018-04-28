# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def main(lambda1, lambda2):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  tvars = tf.trainable_variables()
  l1_regularizer = tf.contrib.layers.l1_regularizer(
      scale=lambda1, scope=None)
  loss_l1 = tf.contrib.layers.apply_regularization(
      l1_regularizer, tvars)
  l2_regularizer = tf.contrib.layers.l2_regularizer(
      scale=lambda2, scope=None)
  loss_l2 = tf.contrib.layers.apply_regularization(
      l2_regularizer, tvars)
  loss = cross_entropy + loss_l1 + loss_l2
  loss = cross_entropy
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})
  return acc

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/datasets/BigLearning/haowen/mnist/inputdata',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  lambda1_set = [0.0001, 0.0003, 0.001]
  lambda2_set = [0.0001, 0.0003, 0.001]
  num1 = len(lambda1_set)
  num2 = len(lambda2_set)
  acc_mat = np.zeros([num1, num2])
  for i in range(num1):
      for j in range(num2):
          lambda1 = lambda1_set[i]
          lambda2 = lambda2_set[j]
          acc = main(lambda1, lambda2)
          acc_mat[i,j] = acc
  print(acc_mat)
