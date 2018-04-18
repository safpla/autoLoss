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
    def __init__(self, config, graph, loss_mode='2'):
        self.config = config
        self.graph = graph
        # ----Loss_mode is only for DEBUG usage.----
        #   0: only mse, 1: mse & l1, 2: mse & l1 & l2
        self.loss_mode = loss_mode
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

    def get_state(self, sess):
        # TODO(haowen) simply concatenate them could cause scale problem

        #state = self.step_number + self.previous_mse_loss\
        #    + self.previous_l1_loss + self.previous_l2_loss\
        #    + self.previous_action
        abs_diff = []
        rel_diff = []
        for v, t in zip(self.previous_valid_loss, self.previous_train_loss):
            abs_diff.append(v - t)
            if t > 1e-6:
                rel_diff.append((v - t) / t)
            else:
                rel_diff.append(0)

        state = (self.previous_valid_loss
                 + self.previous_train_loss
                 + abs_diff
                 + rel_diff)
                 #+ self.previous_mse_loss
                 #+ self.previous_l1_loss
                 #+ self.previous_l2_loss)
        return np.array(state, dtype='f')

    def reset(self):
        """ Reset the model """
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.step_number = [0]
        self.previous_mse_loss = [0] * self.config.num_pre_loss
        self.previous_l1_loss = [0] * self.config.num_pre_loss
        self.previous_l2_loss = [0] * self.config.num_pre_loss
        self.previous_action = [0, 0, 0]
        self.previous_valid_loss = [0] * self.config.num_pre_loss
        self.previous_train_loss = [0] * self.config.num_pre_loss

        # to control when to terminate the episode
        self.endurance = 0
        self.best_loss = 1e10

    def _build_placeholder(self):
        x_size = self.config.dim_input_stud
        with self.graph.as_default():
            self.x_plh = tf.placeholder(shape=[None, x_size], dtype=tf.float32)
            self.y_plh = tf.placeholder(shape=[None], dtype=tf.float32)

    def _build_graph(self):
        h_size = self.config.dim_hidden_stud
        y_size = self.config.dim_output_stud
        lr = self.config.lr_stud

        with self.graph.as_default():
            # ----2-layer ffn----
            #hidden = slim.fully_connected(self.x_plh, h_size,
            #                              activation_fn=tf.nn.tanh)
            #self.pred = slim.fully_connected(hidden, y_size,
            #                                 activation_fn=None)

            # ----quadratic equation----
            #  ---first order---
            x_size = self.config.dim_input_stud
            initial = tf.random_normal(shape=[x_size, 1], stddev=0.1, seed=1)
            w1 = tf.Variable(initial)
            sum1 = tf.matmul(self.x_plh, w1)

            #  ---second order---
            initial = tf.random_normal(shape=[x_size, x_size], stddev=0.01,
                                       seed=1)
            w2 = tf.Variable(initial)
            xx = tf.matmul(tf.reshape(self.x_plh, [-1, x_size, 1]),
                           tf.reshape(self.x_plh, [-1, 1, x_size]))
            sum2 = tf.matmul(tf.reshape(xx, [-1, x_size*x_size]),
                             tf.reshape(w2, [x_size*x_size, 1]))
            # NOTE(Haowen): Divide by 10 is important here to promise
            # convergence.
            self.pred = sum1 + sum2 / 10
            self.w1 = w1
            self.w2 = w2

            # define loss
            self.loss_mse = tf.reduce_mean(tf.square(tf.squeeze(self.pred)
                                                     - self.y_plh))

            # NOTE(Haowen): Somehow the l1,l2 regularizers provided by tf
            # provide a better performance than self-designed regularizers
            # showing in the flowing 6 lines.

            #self.loss_l1 = self.config.lambda1_stud * (
            #    tf.reduce_sum(tf.reduce_sum(tf.abs(w2)))\
            #    + tf.reduce_sum(tf.reduce_sum(tf.abs(w1))))
            #self.loss_l2 = self.config.lambda2_stud * (
            #    tf.reduce_sum(tf.reduce_sum(tf.square(w2)))\
            #    + tf.reduce_sum(tf.reduce_sum(tf.square(w1))))

            tvars = tf.trainable_variables()
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=self.config.lambda1_stud, scope=None)
            self.loss_l1 = tf.contrib.layers.apply_regularization(
                l1_regularizer, tvars)
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=self.config.lambda2_stud, scope=None)
            self.loss_l2 = tf.contrib.layers.apply_regularization(
                l2_regularizer, tvars)
            if self.loss_mode == '0':
                self.loss_total = self.loss_mse
                print('mse loss')
            elif self.loss_mode == '1':
                self.loss_total = self.loss_mse + self.loss_l1
                print('mse loss and l1 loss')
                print('lambda1:', self.config.lambda1_stud)
            else:
                self.loss_total = self.loss_mse + self.loss_l1 + self.loss_l2
                print('mse loss, l1 loss and l2 loss')
                print('lambda1:', self.config.lambda1_stud)
                print('lambda2:', self.config.lambda2_stud)

            # ----Define update operation.----
            self.update_mse = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_mse)
            self.update_l1 = tf.train.GradientDescentOptimizer(lr*1).\
                minimize(self.loss_l1)
            self.update_l2 = tf.train.GradientDescentOptimizer(lr*1).\
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
        loss, _ = sess.run([self.loss_total, self.update_total],
                           feed_dict=feed_dict)
        return loss

    def valid(self, sess, dataset=None):
        """ test on validation set """
        if not dataset:
            dataset = self.valid_dataset
        assert sess.graph is self.graph
        data = dataset.next_batch(dataset.num_examples)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_mse, self.pred, self.y_plh]
        [loss_mse, pred, gdth] = sess.run(fetch, feed_dict=feed_dict)
        loss_mse_np = np.mean(np.square(pred - gdth))
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

        if action[0] == 1:
            # ----Update mse loss.----
            sess.run(self.update_mse, feed_dict=feed_dict)
        elif action[1] == 1:
            # ----Update l1 loss.----
            sess.run(self.update_l1, feed_dict=feed_dict)
        else:
            assert action[2] == 1
            # ----Update l2 loss.----
            sess.run(self.update_l2, feed_dict=feed_dict)
        loss_mse, loss_l1, loss_l2 = sess.run(fetch, feed_dict=feed_dict)
        valid_loss, _, _ = self.valid(sess)
        train_loss, _, _ = self.valid(sess, dataset=self.train_dataset)

        # ----Update state.----
        self.previous_mse_loss = self.previous_mse_loss[1:] + [loss_mse.tolist()]
        self.previous_l1_loss = self.previous_l1_loss[1:] + [loss_l1.tolist()]
        self.previous_l2_loss = self.previous_l2_loss[1:] + [loss_l2.tolist()]
        self.previous_action = action.tolist()
        self.step_number[0] += 1
        self.previous_valid_loss = self.previous_valid_loss[1:]\
            + [valid_loss.tolist()]
        self.previous_train_loss = self.previous_train_loss[1:]\
            + [train_loss.tolist()]

        reward = self.get_step_reward()
        dead = self.check_terminate(sess)
        state = self.get_state(sess)
        return state, reward, dead

    def check_terminate(self, sess):
        # TODO(haowen)
        # Episode terminates on two condition:
        # 1) Convergence: valid loss doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action (not implement yet)
        step = self.step_number[0]
        if step % self.config.valid_frequence_stud == 0:
            self.endurance += 1
            loss, _, _ = self.valid(sess)
            if loss < self.best_loss:
                self.best_step = self.step_number[0]
                self.best_loss = loss
                self.endurance = 0
            if self.endurance > self.config.max_endurance_stud:
                return True
        return False

    def get_step_reward(self):
        # TODO(haowen) Use the decrease of validation loss as step reward
        improve = (self.previous_valid_loss[-2] - self.previous_valid_loss[-1])

        # TODO(haowen) This design of reward may cause unbalance because
        # positive number is more than negative number in nature
        if abs(improve) < 1e-5:
            return 0    # no reward if the difference is too small
        elif improve > 0:
            # TODO(haowen) Try not to give reward to the reduce of loss
            # This reward will strengthen penalty and weaken reward
            return self.config.reward_step_rl
            #return 0
        else:
            return -self.config.reward_step_rl

    def get_final_reward(self, sess):
        assert self.best_loss < 1e10 - 1
        loss_mse = self.best_loss
        reward = self.config.reward_c / loss_mse

        if self.reward_baseline is None:
            self.reward_baseline = reward
        decay = self.config.reward_baseline_decay
        adv = reward - self.reward_baseline
        self.reward_baseline = decay * self.reward_baseline\
            + (1 - decay) * reward
        # TODO(haowen) Try to use maximum instead of shift average as baseline
        # Result: doesn't seem to help too much
        #if self.reward_baseline < reward:
        #    self.reward_baseline = reward
        return reward, adv

    def print_weight(self, sess):
        w1, w2 = sess.run([self.w1, self.w2])
        print('w1: ', w1)
        #print('w2: ', w2)

