""" This module implement a toy task: linear regression """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

from dataio.dataset import Dataset
import utils

logger = utils.get_logger()

class Toy():
    def __init__(self, config, graph):
        self.config = config
        self.graph = graph
        train_data_file = config.train_data_file
        valid_data_file = config.valid_data_file
        self.train_dataset = Dataset()
        self.train_dataset.load_npy(train_data_file)
        self.valid_dataset = Dataset()
        self.valid_dataset.load_npy(valid_data_file)
        self.reset()
        self._build_placeholder()
        self._build_graph()
        self.reward_baseline = None

    def get_state(self):
        """ Return an initial state """
        # TODO(haowen) simply concatenate them could cause scale problem
        state = self.step_number + self.previous_mse_loss\
            + self.previous_l1_loss + self.previous_l2_loss\
            + self.previous_action
        return np.array(state, dtype='f')

    def reset(self):
        """ Reset the model """
        # TODO(haowen) the way to carry step number information should be
        # reconsiderd
        self.step_number = [0]
        self.previous_mse_loss = [0] * self.config.num_pre_loss
        self.previous_l1_loss = [0] * self.config.num_pre_loss
        self.previous_l2_loss = [0] * self.config.num_pre_loss
        self.previous_action = [0, 0, 0]

    def _build_placeholder(self):
        x_size = self.config.dim_input_stud
        self.x_plh = tf.placeholder(shape=[None, x_size], dtype=tf.float32)
        self.y_plh = tf.placeholder(shape=[None], dtype=tf.float32)

    def _build_graph(self):
        h_size = self.config.dim_hidden_stud
        y_size = self.config.dim_output_stud
        lr = self.config.lr_stud

        with self.graph.as_default():
            #hidden = slim.fully_connected(self.x_plh, h_size,
            #                              activation_fn=tf.nn.relu)
            #self.pred = slim.fully_connected(hidden, y_size,
            #                                 activation_fn=tf.nn.softmax)

            hidden = slim.fully_connected(self.x_plh, h_size,
                                          activation_fn=None)
            self.pred = slim.fully_connected(hidden, y_size,
                                             activation_fn=None)

            # define loss
            self.loss_mse = tf.reduce_mean(tf.square(tf.squeeze(self.pred)
                                                     - self.y_plh))
            tvars = tf.trainable_variables()
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=0.0005, scope=None)
            self.loss_l1 = tf.contrib.layers.apply_regularization(
                l1_regularizer, tvars)
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=0.0005, scope=None)
            self.loss_l2 = tf.contrib.layers.apply_regularization(
                l2_regularizer, tvars)
            self.loss_total = self.loss_mse + self.loss_l1 + self.loss_l2

            # define update operation
            self.update_mse = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_mse)
            self.update_l1 = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_l1)
            self.update_l2 = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_l2)
            self.update_total = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_total)
            self.init = tf.global_variables_initializer()

    def train(self, sess):
        """ Optimize mse loss, l1 loss, l2 loss at the same time """
        assert sess.graph is self.graph
        data = self.train_dataset.next_batch(self.config.batch_size)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        #loss, _ = sess.run([self.loss_total, self.update_total],
        #                   feed_dict=feed_dict)
        loss, _ = sess.run([self.loss_mse, self.update_mse],
                           feed_dict=feed_dict)
        return loss

    def valid(self, sess):
        """ test on validation set """
        assert sess.graph is self.graph
        data = self.valid_dataset.next_batch(self.config.num_sample_valid)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_mse, self.pred, self.y_plh]
        [loss_mse, pred, gdth] = sess.run(fetch, feed_dict=feed_dict)
        loss_mse_np = np.mean(np.square(pred - gdth))
        #print('loss_mse_np: ', loss_mse_np)
        return loss_mse, pred, gdth

    def env(self, sess, action):
        """ Given an action, return the new state, reward and whether dead

        Args:
            action: one hot encoding of actions

        Returns:
            state: shape = [dim_state_rl]
            reward: shape = [1]
            dead: boolean
        """
        assert sess.graph is self.graph
        data = self.train_dataset.next_batch(self.config.batch_size)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_mse, self.loss_l1, self.loss_l2]
        loss_mse, loss_l1, loss_l2 = sess.run(fetch, feed_dict=feed_dict)
        # update state
        self.previous_mse_loss = self.previous_mse_loss[1:] + [loss_mse.tolist()]
        self.previous_l1_loss = self.previous_l1_loss[1:] + [loss_l1.tolist()]
        self.previous_l2_loss = self.previous_l2_loss[1:] + [loss_l2.tolist()]
        self.previous_action = action.tolist()
        self.step_number[0] += 1

        if action[0] == 1:
            # update mse loss
            sess.run(self.update_mse, feed_dict=feed_dict)
        elif action[1] == 1:
            # update l1 loss
            sess.run(self.update_l1, feed_dict=feed_dict)
        else:
            assert action[2] == 1
            # update l2 loss
            sess.run(self.update_l2, feed_dict=feed_dict)

        reward = self.get_step_reward()
        dead = self.check_terminate()
        state = self.get_state()
        return state, reward, dead

    def check_terminate(self):
        # TODO(haowen)
        return False

    def get_step_reward(self):
        # TODO(haowen)
        return 0

    def get_final_reward(self, sess):
        loss_mse, _, _ = self.valid(sess)
        valid_ppl = loss_mse / 100
        reward = self.config.reward_c / valid_ppl ** 2

        if self.reward_baseline is None:
            self.reward_baseline = reward
        decay = self.config.reward_baseline_decay
        adv = reward - self.reward_baseline
        self.reward_baseline = decay * self.reward_baseline\
            + (1 - decay) * reward
        return reward, adv

