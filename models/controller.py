""" Decide which loss to update by Reinforce Learning """
# __Author__ == "Haowen Xu"
# __Data__ == "04-07-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import math

class Controller():
    def __init__(self, config, graph):
        self.config = config
        self.graph = graph
        self._build_placeholder()
        # Notice
        #self._build_graph()
        self._build_graph_sigmoid()

    def _build_placeholder(self):
        config = self.config
        s = config.dim_state_rl
        a = config.dim_action_rl
        with self.graph.as_default():
            self.state_plh = tf.placeholder(shape=[None, s], dtype=tf.float32)
            self.reward_plh = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_plh = tf.placeholder(shape=[None, a], dtype=tf.int32)

    def _build_graph(self):
        config = self.config
        h_size = config.dim_hidden_rl
        a_size = config.dim_action_rl
        lr = config.lr_rl
        with self.graph.as_default():
            hidden = slim.fully_connected(self.state_plh, h_size,
                                        biases_initializer=None,
                                        activation_fn=tf.nn.relu)
            self.output = slim.fully_connected(hidden, a_size,
                                            biases_initializer=None,
                                            activation_fn=tf.nn.softmax)
            self.chosen_action = tf.argmax(self.output, 1)
            action = tf.cast(tf.argmax(self.action_plh, 1), tf.int32)
            self.indexes = tf.range(0, tf.shape(self.output)[0])\
                * tf.shape(self.output)[1] + action
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                                self.indexes)
            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)
                                        * self.reward_plh)

            # ----Restore gradients and update them after several iterals.----
            self.tvars = tf.trainable_variables()
            tvars = self.tvars
            self.gradient_plhs = []
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_plh')
                self.gradient_plhs.append(placeholder)

            self.gradients = tf.gradients(self.loss, tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = optimizer.apply_gradients(zip(self.gradient_plhs, tvars))
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def _build_graph_sigmoid(self):
        config = self.config
        h_size = config.dim_hidden_rl
        a_size = config.dim_action_rl
        lr = config.lr_rl
        with self.graph.as_default():
            #k = tf.Variable(10)
            #t = tf.Variable(1)
            self.output = slim.fully_connected(self.state_plh, a_size,
                                            activation_fn=tf.nn.softmax)
            self.chosen_action = tf.argmax(self.output, 1)
            action = tf.cast(tf.argmax(self.action_plh, 1), tf.int32)
            self.indexes = tf.range(0, tf.shape(self.output)[0])\
                * tf.shape(self.output)[1] + action
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                                self.indexes)
            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)
                                        * self.reward_plh)

            # ----Restore gradients and update them after several iterals.----
            self.tvars = tf.trainable_variables()
            tvars = self.tvars
            self.gradient_plhs = []
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_plh')
                self.gradient_plhs.append(placeholder)

            self.gradients = tf.gradients(self.loss, tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = optimizer.apply_gradients(zip(self.gradient_plhs, tvars))
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def get_gradients(self, sess, state, action, reward):
        """ Return the gradients according to one episode

        Args:
            sess: Current tf.Session
            state: shape = [time_steps, dim_state_rl]
            action: shape = [time_steps, dim_action_rl]
            reward: shape = [time_steps]

        Returns:
            grads: Gradients of all trainable variables
        """
        assert sess.graph is self.graph
        feed_dict = {self.reward_plh: reward,
                     self.action_plh: action,
                     self.state_plh: state}
        grads = sess.run(self.gradients, feed_dict=feed_dict)
        return grads

    def sample(self, sess, state, step):
        #
        # Sample an action from a given state, probabilistically

        # Args:
        #     sess: Current tf.Session
        #     state: shape = [dim_state_rl]

        # Returns:
        #     action: shape = [dim_action_rl]
        #
        assert sess.graph is self.graph
        a_dist = sess.run(self.output, feed_dict={self.state_plh: [state]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        action = np.zeros(len(a_dist[0]), dtype='i')
        #print('sample: ', a)

        # ----Free exploring at a certain probability.----
        decay = self.config.explore_rate_decay_rl
        explore_rate = self.config.explore_rate_rl
        explore_rate = explore_rate * math.exp(-step / decay)
        p = np.random.rand(1)
        if p[0] < explore_rate:
            a = np.random.rand(1)
            if a < 1/3:
                a = 0
            elif a < 2/3:
                a = 1
            else:
                a = 2

        # ----Handcraft classifier----
        #f = state[-1]
        #  ---Hard version---
        #  Threshold varies in different tasks, which is related to the SNR
        #  of the data
        #  ------
        #threshold = 1
        #if f < threshold:
        #    a = 0
        #else:
        #    a = 1

        #  ---Soft version---
        #  Equals to one layer one dim ffn with sigmoid activation
        #  ------
        #k = 20
        #t = 1
        #p = 1 / (1 + math.exp(-k * (f - t)))
        #if np.random.rand(1) < p:
        #    a = 1
        #else:
        #    a = 0

        action[a] = 1
        return action

    def load_model(self, sess, checkpoint_dir):
        assert sess.graph is self.graph
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path
        print('loading pretrained model from: ' + model_checkpoint_path)
        self.saver.restore(sess, model_checkpoint_path)

    def save_model(self, sess, global_step):
        assert sess.graph is self.graph
        model_dir = self.config.model_dir
        task_name = 'autoLoss-' + self.config.student_model_name
        task_dir = os.path.join(model_dir, task_name)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        controller = 'ffn'
        student = self.config.student_model_name
        model_name = '{}-{}'.format(controller, student)
        save_path = os.path.join(task_dir, model_name)
        self.saver.save(sess, save_path, global_step=global_step)

    def print_weight(self, sess):
        assert sess.graph is self.graph
        with self.graph.as_default():
            tvars = sess.run(self.tvars)
            for idx, var in enumerate(tvars):
                print('idx:{}, var:{}'.format(idx, var))

