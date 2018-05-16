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
from models import layers
from models.basic_model import Basic_model
from models.gan import Gan
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

logger = utils.get_logger()

class Gan_grid(Gan):
    def __init__(self, config, exp_name='new_exp_gan_grid'):
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
        self.final_hq_baseline = None # average reward over episodes
        self.reset()
        self.fixed_noise_10000 = np.random.normal(size=(10000, config.dim_z))\
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
        self.hq = 0
        self.entropy = 0

        # to control when to terminate the episode
        self.endurance = 0
        self.best_hq = 0
        self.hq_baseline = 0
        self.collapse = False
        self.previous_action = -1
        self.same_action_count = 0

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
            optimizer = tf.train.AdamOptimizer(learning_rate=lr,
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
        best_hq = 0
        hq_baseline = 0
        best_entropy = 0
        endurance = 0
        decay = config.metric_decay
        steps_per_iteration = config.disc_iters + config.gen_iters

        for step in range(0, config.max_training_step, steps_per_iteration):
            # ----Update D network.----
            for i in range(config.disc_iters):
                data = self.train_dataset.next_batch(batch_size)
                x = data['input']
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.real_data: x,
                             self.is_training: True}
                sess.run(self.disc_train_op, feed_dict=feed_dict)

            # ----Update G network.----
            for i in range(config.gen_iters):
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.is_training: True}
                sess.run(self.gen_train_op, feed_dict=feed_dict)

            if step % valid_frequency == 0:
                logger.info('========Step{}========'.format(step))
                logger.info(endurance)
                metrics = self.get_metrics_5x5(num_batches=100)
                logger.info(metrics)
                hq = metrics[0]
                if hq_baseline > 0:
                    hq_baseline = hq_baseline * decay + hq * (1 - decay)
                else:
                    hq_baseline = hq
                logger.info('hq_baseline: {}'.format(hq_baseline))
                self.generate_plot(step)
                endurance += 1
                if hq_baseline > best_hq:
                    best_hq = hq_baseline
                    endurance = 0
                    if save_model:
                        self.save_model(step)

            if step % print_frequency == 0:
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

            if endurance > config.max_endurance_stud:
                break
        logger.info('best_hq: {}'.format(best_hq))

    def get_state(self):
        if self.step_number == 0:
            state = [0] * self.config.dim_state_rl
        else:
            state = [self.step_number / self.config.max_training_step,
                     math.log(self.mag_disc_grad / self.mag_gen_grad),
                     self.ema_gen_cost,
                     (self.ema_disc_cost_real + self.ema_disc_cost_fake) / 2,
                     self.hq,
                     self.entropy / 3.2
                     ]
        return np.array(state, dtype='f')

    def check_terminate(self):
        # TODO(haowen)
        # Early stop and recording the best result
        # Episode terminates on two condition:
        # 1) Convergence: inception score doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action (not implement yet)
        if self.same_action_count > 500:
            logger.info('Terminate reason: Collapse')
            self.collapse = True
            return True
        step = self.step_number
        if step % self.config.valid_frequency_stud == 0:
            self.endurance += 1
            metrics = self.get_metrics_5x5(100)
            hq = metrics[0]
            entropy = metrics[1]
            self.hq = hq
            self.entropy = entropy
            decay = self.config.metric_decay
            if self.hq_baseline > 0:
                self.hq_baseline = self.hq_baseline * decay + hq * (1 - decay)
            else:
                self.hq_baseline = hq
            if self.hq_baseline > self.best_hq:
                logger.info('step: {}, new best result: {}'.\
                            format(step, self.hq_baseline))
                self.best_step = self.step_number
                self.best_hq = self.hq_baseline
                self.best_entropy = entropy
                self.endurance = 0
                self.save_model(step)

        if step % self.config.print_frequency_stud == 0:
            logger.info('----step{}----'.format(step))
            logger.info('hq: {}, entropy: {}'.format(hq, entropy))
            logger.info('hq_baseline: {}'.format(self.hq_baseline))

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
        if self.collapse:
            return 0, -self.config.reward_max_value
        if self.best_entropy < 2.5:
            # lose mode, fail trail
            logger.info('lose mode with entropy: {}'.format(self.best_entropy))
            return 0, -self.config.reward_max_value

        hq = self.best_hq
        reward = self.config.reward_c * hq ** 2
        if self.final_hq_baseline is None:
            self.final_hq_baseline = hq
        baseline_hq = self.final_hq_baseline
        baseline_reward = self.config.reward_c * baseline_hq ** 2
        decay = self.config.inps_baseline_decay
        adv = reward - baseline_reward
        adv = min(adv, self.config.reward_max_value)
        adv = max(adv, -self.config.reward_max_value)
        # ----Shift average----
        self.final_hq_baseline = decay * self.final_hq_baseline\
            + (1 - decay) * hq
        return reward, adv

    def get_metrics_5x5(self, num_batches=100):
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
        hq = np.sum(count_cluster) / (num_batches * batch_size)
        p_cluster = count_cluster / np.sum(count_cluster)
        #print('hq:', np.sum(hq))
        #print('count_cluster:', count_cluster)
        #print('p_cluster:', p_cluster)
        p_cluster += 1e-8
        entropy = -np.sum(p_cluster * np.log(p_cluster))
        #print('entropy:', entropy)
        return hq, entropy

    def get_metrics_2x2(self, num_batches=100):
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
        for i in range(2):
            for j in range(2):
                centers.append([i*1.0-0.5, j*1.0-0.5])
        centers = np.array(centers)
        distance = np.zeros([batch_size*num_batches, 4])
        for i in range(4):
            distance[:,i] = np.sqrt(np.square(all_samples[:,0] - centers[i,0])
                                  + np.square(all_samples[:,1] - centers[i,1]))
        high_quality = distance < config.var_noise * 3
        count_cluster = np.sum(high_quality, 0)
        hq = np.sum(count_cluster) / (num_batches * batch_size)
        p_cluster = count_cluster / np.sum(count_cluster)
        #print('hq:', np.sum(hq))
        print('count_cluster:', count_cluster)
        #print('p_cluster:', p_cluster)
        p_cluster += 1e-8
        entropy = -np.sum(p_cluster * np.log(p_cluster))
        #print('entropy:', entropy)
        return hq, entropy

    def generate_plot(self, step):
        feed_dict = {self.noise: self.fixed_noise_10000,
                     self.is_training: False}
        samples = self.sess.run(self.fake_data, feed_dict=feed_dict)
        task_dir = os.path.join(self.config.save_images_dir, self.exp_name)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        save_path = os.path.join(task_dir, 'images_{}.png'.format(step))

        plt.scatter(samples[:,0], samples[:,1])
        plt.show()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/' + 'gan.cfg')
    config = utils.Parser(config_path)
    gan = Gan(config)
