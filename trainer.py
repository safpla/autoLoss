""" The module for training autoLoss """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import numpy as np
import logging

import models
import utils

logger = utils.get_logger()

class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config):
        self.config = config
        self.g_ctrl = tf.Graph()
        self.g_stud = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess_ctrl = tf.InteractiveSession(config=configProto,
                                               graph=self.g_ctrl)
        self.sess_stud = tf.InteractiveSession(config=configProto,
                                               graph=self.g_stud)
        self.model_ctrl = models.controller.Controller(config, self.g_ctrl)
        if config.student_model_name == 'toy':
            self.model_stud = models.toy.Toy(config, self.g_stud)
        else:
            raise NotImplementedError



    def train(self):
        """ Iteratively training between controller and the multi-loss task """
        config = self.config
        sess_ctrl = self.sess_ctrl
        sess_stud = self.sess.stud
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud

        # initializer both models
        sess_ctrl.run(model_ctrl.init)
        sess_stud.run(model_stud.init)

        # initialize gradient buffer
        gradBuffer = sess_ctrl.run(model_ctrl.tvars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # start episodes
        for ep in range(config.total_episodes):
            state = model_stud.get_initialized_state()
            running_reward = 0
            state_hist = []
            action_hist = []
            reward_hist = []

            # running one episode.
            for i in range(config.max_training_step):
                action = model_ctrl.sample(sess_ctrl, state)
                state_new, reward, dead = self.model_stud.env(action)
                state_hist.append(state)
                action_hist.append(action)
                reward_hist.append(reward)
                state = state_new
                running_reward += reward
                if dead:
                    break

            # update the controller.
            reward_hist = np.array(reward_hist)
            reward_hist = discount_rewards(reward_hist)
            grads = model_ctrl.get_gradients(sess_ctrl, state, action, reward)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if ep % config.update_frequency == 0 and ep != 0:
                feed_dict = dict(zip(model_ctrl.gradient_plhs, gradBuffer))
                _ = sess_ctrl.run(model_ctrl.train_op, feed_dict=feed_dict)
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

            total_reward.append(running_reward)

            if ep % 100 == 0:
                print(np.mean(total_reward[-100:]))







if __name__ == '__main__':
    pass
