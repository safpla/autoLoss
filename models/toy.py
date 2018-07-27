""" This module implement a toy task: linear regression """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

from dataio.dataset import Dataset
import utils
from models.basic_model import Basic_model

logger = utils.get_logger()

def _log(x):
    y = []
    for xx in x:
        y.append(math.log(xx))
    return y

def _normalize1(x):
    y = []
    for xx in x:
        y.append(1 + math.log(xx + 1e-5) / 12)
    return y

def _normalize2(x):
    y = []
    for xx in x:
        y.append(min(1, xx / 20))
    return y

def _normalize3(x):
    y = []
    for xx in x:
        y.append(xx)
    return y


class Toy(Basic_model):
    def __init__(self, config, exp_name='new_exp', loss_mode='1'):
        self.config = config
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        # ----Loss_mode is only for DEBUG usage.----
        #   0: only mse, 1: mse & l1
        self.loss_mode = loss_mode
        self.exp_name = exp_name
        train_data_file = config.train_data_file
        train_stud_data_file = config.train_stud_data_file
        valid_data_file = config.valid_data_file
        self.train_dataset = Dataset()
        self.train_dataset.load_npy(train_data_file)
        self.valid_dataset = Dataset()
        self.valid_dataset.load_npy(valid_data_file)
        self.train_stud_dataset = Dataset()
        self.train_stud_dataset.load_npy(train_stud_data_file)
        self.reset()
        self._build_placeholder()
        self._build_graph()
        self.reward_baseline = None
        self.improve_baseline = None

    def reset(self):
        """ Reset the model """
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.step_number = [0]
        self.previous_mse_loss = [0] * self.config.num_pre_loss
        self.previous_l1_loss = [0] * self.config.num_pre_loss
        self.previous_action = [0, 0]
        self.previous_valid_loss = [0] * self.config.num_pre_loss
        self.previous_train_loss = [0] * self.config.num_pre_loss

        # to control when to terminate the episode
        self.endurance = 0
        self.best_loss = 1e10
        self.improve_baseline = None

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

            # NOTE(Haowen): Somehow the l1 regularizers provided by tf
            # provide a better performance than self-designed regularizers
            # showing in the flowing 6 lines.

            #self.loss_l1 = self.config.lambda1_stud * (
            #    tf.reduce_sum(tf.reduce_sum(tf.abs(w2)))\
            #    + tf.reduce_sum(tf.reduce_sum(tf.abs(w1))))

            tvars = tf.trainable_variables()
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=self.config.lambda1_stud, scope=None)
            self.loss_l1 = tf.contrib.layers.apply_regularization(
                l1_regularizer, tvars)
            if self.loss_mode == '0':
                self.loss_total = self.loss_mse
                print('mse loss')
            elif self.loss_mode == '1':
                self.loss_total = self.loss_mse + self.loss_l1
                print('mse loss and l1 loss')
                print('lambda1:', self.config.lambda1_stud)
            else:
                raise NotImplementedError

            # ----Define update operation.----
            self.update_mse = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_mse)
            self.update_l1 = tf.train.GradientDescentOptimizer(lr*1).\
                minimize(self.loss_l1)
            self.update_total = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_total)
            self.init = tf.global_variables_initializer()

    def train(self):
        """ Optimize mse loss, l1 loss at the same time """
        data = self.train_dataset.next_batch(self.config.batch_size)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        loss, _ = self.sess.run([self.loss_total, self.update_total],
                                feed_dict=feed_dict)
        return loss

    def valid(self, dataset=None):
        """ test on validation set """
        if not dataset:
            dataset = self.valid_dataset
        data = dataset.next_batch(dataset.num_examples)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_mse, self.pred, self.y_plh]
        [loss_mse, pred, gdth] = self.sess.run(fetch, feed_dict=feed_dict)
        loss_mse_np = np.mean(np.square(pred - gdth))
        return loss_mse, pred, gdth

    def response(self, action, mode='TRAIN'):
        """ Given an action, return the new state, reward and whether dead

        Args:
            action: one hot encoding of actions

        Returns:
            state: shape = [dim_state_rl]
            reward: shape = [1]
            dead: boolean
        """
        if mode == 'TRAIN':
            dataset = self.train_dataset
        else:
            dataset = self.train_stud_dataset

        data = dataset.next_batch(self.config.batch_size)
        sess = self.sess
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_mse, self.loss_l1]

        if action[0] == 1:
            # ----Update mse loss.----
            sess.run(self.update_mse, feed_dict=feed_dict)
        elif action[1] == 1:
            # ----Update l1 loss.----
            sess.run(self.update_l1, feed_dict=feed_dict)
        loss_mse, loss_l1 = sess.run(fetch, feed_dict=feed_dict)
        valid_loss, _, _ = self.valid()
        train_loss, _, _ = self.valid(dataset=dataset)

        # ----Update state.----
        self.previous_mse_loss = self.previous_mse_loss[1:] + [loss_mse.tolist()]
        self.previous_l1_loss = self.previous_l1_loss[1:] + [loss_l1.tolist()]
        self.previous_action = action.tolist()
        self.step_number[0] += 1
        self.previous_valid_loss = self.previous_valid_loss[1:]\
            + [valid_loss.tolist()]
        self.previous_train_loss = self.previous_train_loss[1:]\
            + [train_loss.tolist()]

        reward = self.get_step_reward()
        # ----Early stop and record best result.----
        dead = self.check_terminate()
        state = self.get_state()
        return state, reward, dead

    def check_terminate(self):
        # TODO(haowen)
        # Episode terminates on two condition:
        # 1) Convergence: valid loss doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action (not implement yet)
        step = self.step_number[0]
        if step % self.config.valid_frequency_stud == 0:
            self.endurance += 1
            loss, _, _ = self.valid()
            if loss < self.best_loss:
                self.best_step = self.step_number[0]
                self.best_loss = loss
                self.endurance = 0
            if self.endurance > self.config.max_endurance_stud:
                return True
        return False

    def get_step_reward(self):
        # TODO(haowen) Use the decrease of validation loss as step reward
        if self.improve_baseline is None:
            # ----First step, nothing to compare with.----
            improve = 0.1
        else:
            improve = (self.previous_valid_loss[-2] - self.previous_valid_loss[-1])

        # TODO(haowen) Try to use sqrt function instead of sign function
        if self.improve_baseline is None:
            self.improve_baseline = improve
        decay = self.config.reward_baseline_decay
        self.improve_baseline = decay * self.improve_baseline\
            + (1 - decay) * improve

        #TODO(haowen) Remove nonlinearity
        value = math.sqrt(abs(improve) / (abs(self.improve_baseline) + 1e-5))
        #value = abs(improve) / (abs(self.improve_baseline) + 1e-5)
        value = min(value, self.config.reward_max_value)
        return math.copysign(value, improve) * self.config.reward_step_rl

    def get_final_reward(self):
        assert self.best_loss < 1e10 - 1
        loss_mse = self.best_loss
        reward = self.config.reward_c / loss_mse

        if self.reward_baseline is None:
            self.reward_baseline = reward
        decay = self.config.reward_baseline_decay
        adv = reward - self.reward_baseline
        adv = min(adv, self.config.reward_max_value)
        adv = max(adv, -self.config.reward_max_value)
        # TODO(haowen) Try to use maximum instead of shift average as baseline
        # Result: doesn't seem to help too much
        # ----Shift average----
        self.reward_baseline = decay * self.reward_baseline\
            + (1 - decay) * reward
        # ----Maximun----
        #if self.reward_baseline < reward:
        #    self.reward_baseline = reward
        return reward, adv

    def get_state(self):
        abs_diff = []
        rel_diff = []
        if self.improve_baseline is None:
            ib = 1
        else:
            ib = self.improve_baseline

        for v, t in zip(self.previous_valid_loss, self.previous_train_loss):
            abs_diff.append(v - t)
            if t > 1e-6:
                rel_diff.append(v / t)
            else:
                rel_diff.append(1)

        state = ([math.log(rel_diff[-1])] +
                 _normalize1([abs(ib)]) +
                 _normalize2(self.previous_mse_loss[-1:]) +
                 self.previous_l1_loss[-1:]
                 )
        return np.array(state, dtype='f')

