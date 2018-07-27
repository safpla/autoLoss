""" This module implement a gan task """
# __Author__ == "Haowen Xu"
# __Data__ == "04-29-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os

from dataio.dataset_cifar10 import Dataset_cifar10
import utils
from utils import inception_score
from utils import save_images
from models import layers
from models.gan import Gan
logger = utils.get_logger()

class Gan_cifar10(Gan):
    def __init__(self, config, exp_name='new_exp_gan_cifar10'):
        self.config = config
        self.graph = tf.Graph()
        self.exp_name = exp_name
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto,
                                            graph=self.graph)
        self.train_dataset = Dataset_cifar10()
        self.train_dataset.load_cifar10(config.data_dir,
                                        tf.estimator.ModeKeys.TRAIN)
        self.valid_dataset = Dataset_cifar10()
        self.valid_dataset.load_cifar10(config.data_dir,
                                        tf.estimator.ModeKeys.EVAL)

        self._build_placeholder()
        self._build_graph()
        self.final_inps_baseline = None # ema of final_inps over episodes
        # ema of metrics track over episodes
        n_steps = int(config.max_training_step / config.print_frequency_stud)
        self.metrics_track_baseline = -np.ones([n_steps])
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
        self.mag_gen_grad = None
        self.mag_disc_grad = None
        self.inception_score = 0
        self.previous_action = -1
        self.same_action_count = 0
        self.task_dir = None

        # to control when to terminate the episode
        self.endurance = 0
        self.best_inception_score = 0
        self.inps_baseline = 0
        self.collapse = False

    def _build_placeholder(self):
        with self.graph.as_default():
            dim_x = self.config.dim_x
            dim_z = self.config.dim_z
            self.real_data = tf.placeholder(tf.int32, shape=[None, dim_x],
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
            real_data_t = 2*((tf.cast(self.real_data, tf.float32)/255.)-.5)
            real_data_NCHW = tf.reshape(real_data_t, [-1, 3, 32, 32])
            real_data_NHWC = tf.transpose(real_data_NCHW, perm=[0, 2, 3, 1])
            real_data = tf.reshape(real_data_NHWC, [-1, dim_x])
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

            self.grads = gen_grad + disc_grad

    def generator(self, input):
        dim_z = self.config.dim_z
        dim_c = self.config.dim_c
        with tf.variable_scope('Generator'):
            output = layers.linear(input, 4*4*4*dim_c, name='LN1')
            output = layers.batchnorm(output, is_training=self.is_training,
                                      name='BN1')
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4, 4, 4*dim_c])

            output_shape = [-1, 8, 8, 2*dim_c]
            output = layers.deconv2d(output, output_shape, name='Deconv2')
            output = layers.batchnorm(output, is_training=self.is_training,
                                      name='BN2')
            output = tf.nn.relu(output)

            output_shape = [-1, 16, 16, dim_c]
            output = layers.deconv2d(output, output_shape, name='Decovn3')
            output = layers.batchnorm(output, is_training=self.is_training,
                                      name='BN3')
            output = tf.nn.relu(output)

            output_shape = [-1, 32, 32, 3]
            output = layers.deconv2d(output, output_shape, name='Deconv4')
            output = tf.nn.tanh(output)

            return tf.reshape(output, [-1, 32*32*3])

    def discriminator(self, inputs, reuse=False):
        dim_c = self.config.dim_c
        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 32, 32, 3])

            output = layers.conv2d(output, dim_c, name='Conv1')
            output = tf.nn.leaky_relu(output)

            output = layers.conv2d(output, 2*dim_c, name='Conv2')
            output = layers.batchnorm(output, is_training=self.is_training,
                                      name='BN2')
            output = tf.nn.leaky_relu(output)

            output = layers.conv2d(output, 4*dim_c, name='Conv3')
            output = layers.batchnorm(output, is_training=self.is_training,
                                      name='BN3')
            output = tf.nn.leaky_relu(output)

            output = tf.reshape(output, [-1, 4*4*4*dim_c])
            output = layers.linear(output, 1, name='LN4')

            return tf.reshape(output, [-1])

    def get_state(self):
        if self.step_number == 0:
            state = [0] * self.config.dim_state_rl
        else:
            state = [self.step_number / self.config.max_training_step,
                     math.log(self.mag_disc_grad / self.mag_gen_grad),
                     self.ema_gen_cost,
                     (self.ema_disc_cost_real + self.ema_disc_cost_fake) / 2,
                     self.inception_score / 10 * 7 / 9,
                     ]
        return np.array(state, dtype='f')

    def get_inception_score(self, num_batches, splits=None):
        all_samples = []
        config = self.config
        if not splits:
            splits = config.inps_splits
        batch_size = 100
        dim_z = config.dim_z
        for i in range(num_batches):
            z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
            feed_dict = {self.noise: z, self.is_training: False}
            samples = self.sess.run(self.fake_data, feed_dict=feed_dict)
            all_samples.append(samples)
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*255./2.).astype(np.int32)
        all_samples = all_samples.reshape((-1, 32, 32, 3))
        return inception_score.get_inception_score(list(all_samples),
                                                   splits=splits)

    def generate_images(self, step):
        feed_dict = {self.noise: self.fixed_noise_128,
                     self.is_training: False}
        samples = self.sess.run(self.fake_data, feed_dict=feed_dict)
        samples = ((samples+1.)*255./2.).astype('int32')
        task_dir = os.path.join(self.config.save_images_dir, self.exp_name)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        save_path = os.path.join(task_dir, 'images_{}.jpg'.format(step))
        save_images.save_images(samples.reshape((-1, 32, 32, 3)), save_path)


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/' + 'gan.cfg')
    config = utils.Parser(config_path)
    gan = Gan(config)
