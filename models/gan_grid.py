""" This module implement a gan task """
# __Author__ == "Haowen Xu"
# __Data__ == "04-29-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import time
import os

from dataio.dataset import Dataset
import utils
from utils import inception_score
from utils import save_images
from models import layers
from models.basic_model import Basic_model
from models.gan import Gan

logger = utils.get_logger()

class Gan_grid(Gan):
    def __init__(self, config, exp_name='new_exp'):
        self.config = config
        self.graph = tf.Graph()
        self.exp_name = exp_name
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        # ----Loss_mode is only for DEBUG usage.----
        self.train_dataset = Dataset()
        self.train_dataset.load_npy(config.train_data_file)
        self.valid_dataset = Dataset()
        self.valid_dataset.load_npy(config.valid_data_file)
        self._build_placeholder()
        self._build_graph()
        self.reward_baseline = None # average reward over episodes
        self.reset()
        self.fixed_noise_128 = np.random.normal(size=(128, config.dim_z))\
            .astype('float32')

    def reset(self):
        # ----Reset the model.----
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.step_number = 0
        self.ema_gen_cost = None
        self.ema_disc_cost_real = None
        self.ema_disc_cost_fake = None
        self.prst_gen_cost = None
        self.prst_disc_cost_real = None
        self.prst_disc_cost_fake = None
        self.hq_ratio = 0
        self.entropy = 0

        # to control when to terminate the episode
        self.endurance = 0
        self.best_hq_ratio = 0

    def _build_placeholder(self):
        with self.graph.as_default():
            dim_x = self.config.dim_x
            dim_z = self.config.dim_z
            bs = self.config.batch_size
            self.real_data = tf.placeholder(tf.float32, shape=[None, dim_x],
                                            name='real_data')
            self.noise = tf.placeholder(tf.float32, shape=[None, dim_z],
                                        name='noise')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.lr_plh = tf.placeholder(dtype=tf.float32)

    def _build_graph(self):
        dim_x = self.config.dim_x
        dim_z = self.config.dim_z
        batch_size = self.config.batch_size
        lr = self.config.lr_stud
        beta1 = self.config.beta1
        beta2 = self.config.beta2

        with self.graph.as_default():
            real_data = self.real_data
            fake_data = self.generator(self.noise)
            disc_real = self.discriminator(real_data)
            disc_fake = self.discriminator(fake_data, reuse=True)

            gen_cost = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=disc_fake, labels=tf.ones_like(disc_fake)
                )
            )
            disc_cost_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=disc_fake, labels=tf.zeros_like(disc_fake)
                )
            )
            disc_cost_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=disc_real, labels=tf.ones_like(disc_real)
                )
            )
            disc_cost = (disc_cost_fake + disc_cost_real) / 2.

            tvars = tf.trainable_variables()
            gen_tvars = [v for v in tvars if 'Generator' in v.name]
            disc_tvars = [v for v in tvars if 'Discriminator' in v.name]

            gen_grad = tf.gradients(gen_cost, gen_tvars)
            disc_grad = tf.gradients(disc_cost, disc_tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_plh,
                                                beta1=beta1,
                                                beta2=beta2)
            gen_train_op = optimizer.apply_gradients(
                zip(gen_grad, gen_tvars))
            disc_train_op = optimizer.apply_gradients(
                zip(disc_grad, disc_tvars))

            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()
            self.fake_data = fake_data
            self.gen_train_op = gen_train_op
            self.disc_train_op = disc_train_op
            self.update = [gen_train_op, disc_train_op]

            self.gen_cost = gen_cost
            self.gen_grad = gen_grad
            self.gen_tvars = gen_tvars

            self.disc_cost_fake = disc_cost_fake
            self.disc_cost_real = disc_cost_real
            self.disc_grad = disc_grad
            self.disc_tvars = disc_tvars
            self.disc_real = disc_real
            self.disc_fake = disc_fake

    def generator(self, input):
        dim_x = self.config.dim_x
        n_hidden = self.config.n_hidden_gen
        with tf.variable_scope('Generator'):
            output = layers.linear(input, n_hidden, name='LN1', stdev=0.2)
            output = layers.batchnorm(output, is_training=self.is_training,
                                      name='BN1')
            output = tf.nn.relu(output)
            output = layers.linear(output, n_hidden, name='LN2', stdev=0.2)
            output = layers.batchnorm(output, is_training=self.is_training,
                                      name='BN2')
            output = tf.nn.relu(output)
            output = layers.linear(output, dim_x, name='LN3', stdev=0.2)
            #output = slim.fully_connected(input, n_hidden,
            #                              activation_fn=tf.nn.relu)
            #output = slim.fully_connected(output, n_hidden,
            #                              activation_fn=tf.nn.relu)
            #output = slim.fully_connected(output, dim_x, activation_fn=None)
            return output

    def discriminator(self, input, reuse=False):
        n_hidden = self.config.n_hidden_disc
        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            output = layers.linear(input, n_hidden, name='LN1', stdev=0.2)
            output = tf.nn.relu(output)
            output = layers.linear(output, 1, name='LN2', stdev=0.2)
            #output = slim.fully_connected(input, n_hidden,
            #                              activation_fn=tf.nn.relu)
            #output = slim.fully_connected(output, 1, activation_fn=None)
            return tf.reshape(output, [-1])

    def train(self, save_model=False):
        sess = self.sess
        config = self.config
        batch_size = config.batch_size
        dim_z = config.dim_z
        valid_frequency = config.valid_frequency_stud
        print_frequency = config.print_frequency_stud
        best_hq_ratio = 0
        best_entropy = 0
        endurance = 0
        lr = config.lr_stud

        for step in range(config.max_training_step):
            # ----Update D network.----
            if lr > config.lr_stud * 0.1:
                lr = lr * config.lr_decay_stud
            for i in range(config.disc_iters):
                data = self.train_dataset.next_batch(batch_size)
                x = data['input']
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.real_data: x,
                             self.is_training: True,
                             self.lr_plh: lr}
                fetch = [self.fake_data, self.disc_train_op]
                sess.run(self.disc_train_op, feed_dict=feed_dict)

            # ----Update G network.----
            for i in range(config.gen_iters):
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.is_training: True,
                             self.lr_plh: lr}
                sess.run(self.gen_train_op, feed_dict=feed_dict)

            if step % valid_frequency == (valid_frequency - 1):
                logger.info('========Step{}========'.format(step + 1))
                metrics = self.get_metrics(num_batches=100)
                hq_ratio = metrics[0]
                if hq_ratio > best_hq_ratio:
                    best_hq_ratio = hq_ratio
                    endurance = 0
                endurance += 1
                if endurance > config.max_endurance_stud:
                    break

                #self.generate_plot(step)

            if step % print_frequency == (print_frequency - 1):
                data = self.train_dataset.next_batch(batch_size)
                x = data['input']
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.real_data: x,
                             self.is_training: False}
                fetch = [self.gen_cost,
                         self.disc_cost_fake,
                         self.disc_cost_real]
                r = sess.run(fetch, feed_dict=feed_dict)
                logger.info('gen_cost: {}'.format(r[0]))
                logger.info('disc_cost fake: {}, real: {}'.format(r[1], r[2]))
                logger.info('lr: {}'.format(lr))
        print('best_hq_ratio: {}'.format(best_hq_ratio))

    def response(self, action):
        # Given an action, return the new state, reward and whether dead

        # Args:
        #     action: one hot encoding of actions

        # Returns:
        #     state: shape = [dim_state_rl]
        #     reward: shape = [1]
        #     dead: boolean
        #
        sess = self.sess
        batch_size = self.config.batch_size
        dim_z = self.config.dim_z
        alpha = self.config.state_decay
        lr = self.config.lr_stud

        data = self.train_dataset.next_batch(batch_size)
        x = data['input']
        z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
        feed_dict = {self.noise: z, self.real_data: x,
                     self.is_training: True,
                     self.lr_plh: lr}
        a = np.argmax(np.array(action))
        sess.run(self.update[a], feed_dict=feed_dict)

        fetch = [self.gen_cost, self.disc_cost_real, self.disc_cost_fake]
        r = sess.run(fetch, feed_dict=feed_dict)
        gen_cost = r[0]
        disc_cost_real = r[1]
        disc_cost_fake = r[2]
        self.prst_gen_cost = r[0]
        self.prst_disc_cost_real = r[1]
        self.prst_disc_cost_fake = r[2]

        # ----Update state.----
        self.step_number += 1
        if self.ema_gen_cost is None:
            self.ema_gen_cost = gen_cost
            self.ema_disc_cost_real = disc_cost_real
            self.ema_disc_cost_fake = disc_cost_fake
        else:
            self.ema_gen_cost = self.ema_gen_cost * alpha\
                + gen_cost * (1 - alpha)
            self.ema_disc_cost_real = self.ema_disc_cost_real * alpha\
                + disc_cost_real * (1 - alpha)
            self.ema_disc_cost_fake = self.ema_disc_cost_fake * alpha\
                + disc_cost_fake * (1 - alpha)


        reward = self.get_step_reward()
        # ----Early stop and record best result.----
        dead = self.check_terminate()
        state = self.get_state()
        return state, reward, dead

    def get_state(self):
        if self.step_number == 0:
            state = [0] * self.config.dim_state_rl
        else:
            state = [self.step_number / self.config.max_training_step,
                     self.ema_gen_cost / 5,
                     self.ema_disc_cost_real,
                     self.ema_disc_cost_fake,
                     self.hq_ratio,
                     self.entropy / 3.2
                     ]
        return np.array(state, dtype='f')

    def check_terminate(self):
        # TODO(haowen)
        # Early stop and recording the best result
        # Episode terminates on two condition:
        # 1) Convergence: inception score doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action (not implement yet)
        step = self.step_number
        if step % self.config.valid_frequency_stud == 0:
            self.endurance += 1
            metrics = self.get_metrics(100)
            hq_ratio = metrics[0]
            entropy = metrics[1]
            self.hq_ratio = hq_ratio
            self.entropy = entropy
            logger.info('----step{}----'.format(step))
            logger.info('hq_ratio: {}'.format(hq_ratio))
            logger.info('entropy: {}'.format(entropy))
            if hq_ratio > self.best_hq_ratio:
                self.best_step = self.step_number
                self.best_hq_ratio = hq_ratio
                self.best_entropy = entropy
                self.endurance = 0

        if step > self.config.max_training_step:
            return True
        if self.config.stop_strategy_stud == 'prescribed_steps':
            pass
        elif self.config.stop_strategy_stud == 'exceeding_endurance' and \
                self.endurance > self.config.max_endurance_stud:
            return True
        return False

    def get_step_reward(self):
        return 0

    def get_final_reward(self):
        if self.best_entropy < 2.5:
            # lose mode, fail trail
            logger.info('lose mode with entropy: {}'.format(self.best_entropy))
            return 0, -20
        reward = self.best_hq_ratio * self.config.reward_c
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

    def get_metrics(self, num_batches=100):
        all_samples = []
        config = self.config
        batch_size = 100
        dim_z = config.dim_z
        for i in range(num_batches):
            z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
            feed_dict = {self.noise: z, self.is_training: False}
            samples = self.sess.run(self.fake_data, feed_dict=feed_dict)
            all_samples.append(samples)
        all_samples = np.concatenate(all_samples, axis=0)

        centers = []
        for i in range(5):
            for j in range(5):
                centers.append([i*0.5-1, j*0.5-1])
        centers = np.array(centers)
        distance = np.zeros([batch_size*num_batches, 25])
        for i in range(25):
            distance[:,i] = np.sqrt(np.square(all_samples[:,0] - centers[i,0])
                                  + np.square(all_samples[:,1] - centers[i,1]))
        high_quality = distance < config.var_noise * 3
        count_cluster = np.sum(high_quality, 0)
        hq_ratio = np.sum(count_cluster) / (num_batches * batch_size)
        p_cluster = count_cluster / np.sum(count_cluster)
        #print('hq_ratio:', np.sum(hq_ratio))
        #print('count_cluster:', count_cluster)
        #print('p_cluster:', p_cluster)
        p_cluster += 1e-8
        entropy = -np.sum(p_cluster * np.log(p_cluster))
        #print('entropy:', entropy)
        return hq_ratio, entropy


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/' + 'gan.cfg')
    config = utils.Parser(config_path)
    gan = Gan(config)
