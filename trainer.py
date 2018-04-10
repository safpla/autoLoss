""" The module for training autoLoss """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import numpy as np
import logging
import os

from models import controller
from models import toy
import utils


logger = utils.get_logger()

def discount_rewards(reward):
    # TODO(haowen) simply use final reward as step reward, refine it later
    reward_dis = np.array([reward[-1]] * len(reward))
    return reward_dis

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
        self.model_ctrl = controller.Controller(config, self.g_ctrl)
        if config.student_model_name == 'toy':
            self.model_stud = toy.Toy(config, self.g_stud)
        else:
            raise NotImplementedError

    def train(self):
        """ Iteratively training between controller and the multi-loss task """
        config = self.config
        sess_ctrl = self.sess_ctrl
        sess_stud = self.sess_stud
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        total_reward = []

        # initializer controllor
        sess_ctrl.run(model_ctrl.init)

        # initialize gradient buffer
        gradBuffer = sess_ctrl.run(model_ctrl.tvars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # start episodes
        for ep in range(config.total_episodes):
            # initializer student / environment
            logger.info('=================')
            logger.info('episodes: {}'.format(ep))

            sess_stud.run(model_stud.init)
            model_stud.reset()

            state = model_stud.get_state()
            running_reward = 0
            state_hist = []
            action_hist = []
            reward_hist = []

            # running one episode.
            for i in range(config.max_training_step):
                #logger.info('----train_step: {}----'.format(i))
                action = model_ctrl.sample(sess_ctrl, state)
                if i < 10:
                    logger.info('sampling an action: {}'.format(action))
                state_new, reward, dead = model_stud.env(sess_stud, action)
                #logger.info('current mse loss: {}'.format(state_new[10]))
                #logger.info('current l1 loss: {}'.format(state_new[20]))
                #logger.info('current l2 loss: {}'.format(state_new[30]))
                #logger.info('reward: {}'.format(reward))
                #logger.info('dead: {}'.format(dead))
                state_hist.append(state)
                action_hist.append(action)
                reward_hist.append(reward)
                state = state_new
                running_reward += reward
                if dead:
                    break
            final_reward, adv = model_stud.get_final_reward(self.sess_stud)
            running_reward += final_reward
            reward_hist[-1] = adv * 10
            logger.info('final_reward: {}'.format(final_reward))
            # update the controller.
            reward_hist = np.array(reward_hist)
            reward_hist = discount_rewards(reward_hist)
            grads = model_ctrl.get_gradients(sess_ctrl, state_hist,
                                             action_hist, reward_hist)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if ep % config.update_frequency == 0 and ep != 0:
                logger.info('UPDATE CONTROLLOR')
                feed_dict = dict(zip(model_ctrl.gradient_plhs, gradBuffer))
                _ = sess_ctrl.run(model_ctrl.train_op, feed_dict=feed_dict)
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

            total_reward.append(running_reward)

            if ep % 10 == 0:
                print(np.mean(total_reward[-10:]))


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/regression.cfg')
    config = utils.Parser(config_path)
    trainer = Trainer(config)
    trainer.train()
