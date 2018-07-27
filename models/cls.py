""" This module implement a toy task: linear classification """
# __Author__ == "Haowen Xu"
# __Data__ == "04-25-2018"

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
        y.append(math.log(xx + 1) / 10)
        #y.append(xx)
    return y

def weight_variable(shape, name='b'):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name='b'):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Cls(Basic_model):
    def __init__(self, config, exp_name='new_exp', loss_mode='1'):
        self.config = config
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        self.exp_name = exp_name
        self.loss_mode = loss_mode
        train_data_file = config.train_data_file
        valid_data_file = config.valid_data_file
        test_data_file = config.test_data_file
        train_stud_data_file = config.train_stud_data_file
        self.train_dataset = Dataset()
        self.train_dataset.load_npy(train_data_file)
        self.valid_dataset = Dataset()
        self.valid_dataset.load_npy(valid_data_file)
        self.test_dataset = Dataset()
        self.test_dataset.load_npy(test_data_file)
        self.train_stud_dataset = Dataset()
        self.train_stud_dataset.load_npy(train_stud_data_file)
        self.reset()
        self._build_placeholder()
        self._build_graph()
        self.reward_baseline = None # average reward over episodes
        self.improve_baseline = None # averge improvement over steps

    def reset(self):
        # ----Reset the model.----
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.step_number = [0]
        self.previous_ce_loss = [0] * self.config.num_pre_loss
        self.previous_l1_loss = [0] * self.config.num_pre_loss
        self.previous_valid_acc = [0] * self.config.num_pre_loss
        self.previous_train_acc = [0] * self.config.num_pre_loss
        self.previous_action = [0] * self.config.dim_action_rl
        self.previous_valid_loss = [0] * self.config.num_pre_loss
        self.previous_train_loss = [0] * self.config.num_pre_loss
        self.task_dir = None

        # to control when to terminate the episode
        self.endurance = 0
        self.best_loss = 1e10
        self.best_acc = 0
        self.test_acc = 0
        self.improve_baseline = None

    def _build_placeholder(self):
        x_size = self.config.dim_input_stud
        with self.graph.as_default():
            self.x_plh = tf.placeholder(shape=[None, x_size], dtype=tf.float32)
            self.y_plh = tf.placeholder(shape=[None], dtype=tf.int32)

    def _build_graph(self):
        x_size = self.config.dim_input_stud
        h_size = self.config.dim_hidden_stud
        y_size = self.config.dim_output_stud
        lr = self.config.lr_stud

        with self.graph.as_default():
            # ----3-layer ffn----
            #hidden1 = slim.fully_connected(self.x_plh, h_size,
            #                              activation_fn=tf.nn.tanh)
            #hidden2 = slim.fully_connected(hidden1, 32,
            #                               activation_fn=tf.nn.tanh)
            #self.pred = slim.fully_connected(hidden1, y_size,
            #                                 activation_fn=None)

            w1 = weight_variable([x_size, h_size], name='w1')
            b1 = bias_variable([h_size], name='b1')
            hidden = tf.nn.relu(tf.matmul(self.x_plh, w1) + b1)

            w2 = weight_variable([h_size, y_size], name='w2')
            b2 = bias_variable([y_size], name='b2')
            self.pred = tf.matmul(hidden, w2) + b2

            # define loss
            self.loss_ce = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_plh,
                    logits=self.pred,
                    name='loss')
            )
            y_ = tf.argmax(self.pred, 1, output_type=tf.int32)
            correct_prediction = tf.equal(y_, self.y_plh)
            self.correct_prediction = correct_prediction
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                   tf.float32))

            tvars = tf.trainable_variables()
            self.tvars = tvars
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=self.config.lambda1_stud, scope=None)
            self.loss_l1 = tf.contrib.layers.apply_regularization(
                l1_regularizer, tvars)
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=self.config.lambda2_stud, scope=None)
            self.loss_l2 = tf.contrib.layers.apply_regularization(
                l2_regularizer, tvars)
            if self.loss_mode == '0':
                self.loss_total = self.loss_ce
                print('ce loss')
            elif self.loss_mode == '1':
                self.loss_total = self.loss_ce + self.loss_l1
                print('ce loss and l1 loss')
                print('lambda1:', self.config.lambda1_stud)
            elif self.loss_mode == '2':
                self.loss_total = self.loss_ce + self.loss_l2
                print('ce loss and l2 loss')
                print('lambda2:', self.config.lambda2_stud)
            elif self.loss_mode == '3':
                self.loss_total = self.loss_ce + self.loss_l1 + self.loss_l2
                print('ce loss, l1 loss and l2 loss')
                print('lambda1:', self.config.lambda1_stud)
                print('lambda2:', self.config.lambda2_stud)
            else:
                raise NotImplementedError

            # ----Define update operation.----
            self.update_ce = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_ce)
            self.update_l1 = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_l1)
            self.update_l2 = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_l2)
            self.update_total = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_total)
            self.update = [self.update_ce, self.update_l1, self.update_l2]
            self.init = tf.global_variables_initializer()

    def train(self):
        # ----Optimize total loss.----
        data = self.train_dataset.next_batch(self.config.batch_size)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_ce, self.accuracy, self.update_total]
        loss, acc, _ = self.sess.run(fetch, feed_dict=feed_dict)
        return loss, acc

    def valid(self, dataset=None):
        # ----Test on validation set.----
        if not dataset:
            dataset = self.valid_dataset
        data = dataset.next_batch(dataset.num_examples)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_ce, self.accuracy, self.pred, self.y_plh]
        [loss_ce, acc, pred, gdth] = self.sess.run(fetch, feed_dict=feed_dict)
        return loss_ce, acc, pred, gdth

    def response(self, action, mode='TRAIN'):
        # Given an action, return the new state, reward and whether dead

        # Args:
        #     action: one hot encoding of actions

        # Returns:
        #     state: shape = [dim_state_rl]
        #     reward: shape = [1]
        #     dead: boolean
        #
        sess = self.sess
        if mode == 'TRAIN':
            dataset = self.train_dataset
        else:
            dataset = self.train_dataset
        data = dataset.next_batch(self.config.batch_size)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}

        a = np.argmax(np.array(action))
        sess.run(self.update[a], feed_dict=feed_dict)
        fetch = [self.loss_ce, self.loss_l1]
        loss_ce, loss_l1 = sess.run(fetch, feed_dict=feed_dict)
        valid_loss, valid_acc, _, _ = self.valid()
        train_loss, train_acc, _, _ = self.valid(dataset=dataset)

        # ----Update state.----
        self.previous_ce_loss = self.previous_ce_loss[1:] + [loss_ce.tolist()]
        self.previous_l1_loss = self.previous_l1_loss[1:] + [loss_l1.tolist()]
        self.previous_action = action.tolist()
        self.step_number[0] += 1
        self.previous_valid_loss = self.previous_valid_loss[1:]\
            + [valid_loss.tolist()]
        self.previous_train_loss = self.previous_train_loss[1:]\
            + [train_loss.tolist()]
        self.previous_valid_acc = self.previous_valid_acc[1:]\
            + [valid_acc.tolist()]
        self.previous_train_acc = self.previous_train_acc[1:]\
            + [train_acc.tolist()]

        reward = self.get_step_reward()
        # ----Early stop and record best result.----
        dead = self.check_terminate()
        state = self.get_state()
        return state, reward, dead

    def check_terminate(self):
        # TODO(haowen)
        # Early stop and recording the best result
        # Episode terminates on two condition:
        # 1) Convergence: valid loss doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action (not implement yet)
        step = self.step_number[0]
        if step % self.config.valid_frequency_stud == 0:
            self.endurance += 1
            loss, acc, _, _ = self.valid()
            if acc > self.best_acc:
                self.best_step = self.step_number[0]
                self.best_loss = loss
                self.best_acc = acc
                self.endurance = 0
                _, test_acc, _, _= self.valid(dataset=self.test_dataset)
                self.test_acc = test_acc
            if self.endurance > self.config.max_endurance_stud:
                return True
        return False

    def get_step_reward(self):
        # TODO(haowen) Use the decrease of validation loss as step reward
        if self.improve_baseline is None:
            # ----First step, nothing to comparing with.----
            improve = 0.1
        else:
            improve = (self.previous_valid_loss[-2] - self.previous_valid_loss[-1])

        # TODO(haowen) Try to use sqrt function instead of sign function
        # ----With baseline.----
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
        # ----Without baseline.----
        #return math.copysign(math.sqrt(abs(improve)), improve)

        # TODO(haowen) This design of reward may cause unbalance because
        # positive number is more than negative number in nature
        #if abs(improve) < 1e-5:
        #    return 0    # no reward if the difference is too small
        #elif improve > 0:
        #    # TODO(haowen) Try not to give reward to the reduce of loss
        #    # This reward will strengthen penalty and weaken reward
        #    return self.config.reward_step_rl
        #    #return 0
        #else:
        #    return -self.config.reward_step_rl

    def get_final_reward(self):
        acc = max(self.best_acc, 1 / self.config.dim_output_stud)
        reward = -self.config.reward_c / acc

        if self.reward_baseline is None:
            self.reward_baseline = reward
        decay = self.config.reward_baseline_decay
        adv = reward - self.reward_baseline
        adv = min(adv, self.config.reward_max_value)
        adv = max(adv, -self.config.reward_max_value)
        # ----Shift average----
        self.reward_baseline = decay * self.reward_baseline\
            + (1 - decay) * reward
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
            state = (rel_diff[-1:] +
                     _normalize1([abs(ib)]) +
                     _normalize2(self.previous_ce_loss[-1:]) +
                     _normalize3(self.previous_l1_loss[-1:]) +
                     self.previous_train_acc[-1:] +
                     [self.previous_train_acc[-1] - self.previous_valid_acc[-1]] +
                     self.previous_valid_acc[-1:]
                 )
        return np.array(state, dtype='f')

