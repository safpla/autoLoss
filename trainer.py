""" The module for training autoLoss """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys

from models import controller
from models import toy
import utils
from utils.analyse_utils import loss_analyzer


logger = utils.get_logger()

def discount_rewards(reward, final_reward):
    # TODO(haowen) Final reward + step reward
    reward_dis = np.array(reward) + np.array(final_reward)
    return reward_dis

class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config):
        self.config = config
        self.g_ctrl = tf.Graph()
        #self.g_ctrl.device('/gpu:0')
        self.g_stud = tf.Graph()
        #self.g_stud.device('/gpu:0')
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess_ctrl = tf.InteractiveSession(config=configProto,
                                               graph=self.g_ctrl)
        self.model_ctrl = controller.Controller(config, self.g_ctrl)
        if config.student_model_name == 'toy':
            self.model_stud = toy.Toy(config, self.g_stud)
        else:
            raise NotImplementedError

    def train(self, load_ctrl=None):
        """ Iteratively training between controller and the multi-loss task """
        config = self.config
        sess_ctrl = self.sess_ctrl
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        total_reward = []
        best_reward = 0
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        lr = config.lr_rl

        # ----Initializer controllor.----
        sess_ctrl.run(model_ctrl.init)
        if load_ctrl:
            model_ctrl.load_model(sess_ctrl, load_ctrl)

        # ----Initialize gradient buffer.----
        gradBuffer = sess_ctrl.run(model_ctrl.tvars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # ----Start episodes.----
        for ep in range(config.total_episodes):
            # initializer student / environment
            logger.info('=================')
            logger.info('episodes: {}'.format(ep))

            self.sess_stud = tf.InteractiveSession(config=configProto,
                                                graph=self.g_stud)
            sess_stud = self.sess_stud
            sess_stud.run(model_stud.init)
            model_stud.reset()

            state = model_stud.get_state(sess_stud)
            running_reward = 0
            state_hist = []
            action_hist = []
            reward_hist = []
            valid_loss_hist = []
            train_loss_hist = []
            old_action = []

            # ----Running one episode.----
            for i in range(config.max_training_step):
                action = model_ctrl.sample(sess_ctrl, state, step=ep)
                state_new, reward, dead = model_stud.env(sess_stud, action)
                state_hist.append(state)
                action_hist.append(action)
                reward_hist.append(reward)
                valid_loss_hist.append(model_stud.previous_valid_loss[-1])
                train_loss_hist.append(model_stud.previous_train_loss[-1])

                #if i % 1 == 0 and i > -1:
                #    logger.info('----train_step: {}----'.format(i))
                #    logger.info('state:{}'.format(state_new))
                #    logger.info('action: {}'.format(action))
                #    logger.info('reward:{}'.format(reward))
                #    l = model_stud.previous_valid_loss
                #    logger.info('loss_imp: {}'.format(l[-2] - l[-1]))
                #    logger.info('train_loss: {}'.format(state_new[3]))
                #    logger.info('valid_loss: {}'.format(l[-1]))
                #    model_stud.print_weight(sess_stud)

                old_action = action
                state = state_new
                running_reward += reward
                if dead:
                    break

            # ----Only use the history before the best result.----
            state_hist = state_hist[:model_stud.best_step]
            action_hist = action_hist[:model_stud.best_step]
            reward_hist = reward_hist[:model_stud.best_step]
            valid_loss_hist = valid_loss_hist[:model_stud.best_step + 1]
            train_loss_hist = train_loss_hist[:model_stud.best_step + 1]

            final_reward, adv = model_stud.get_final_reward(sess_stud)
            loss = model_stud.best_loss
            running_reward += final_reward
            logger.info('final_reward: {}'.format(final_reward))
            logger.info('loss: {}'.format(loss))
            logger.info('adv: {}'.format(adv))

            # ----Update the controller.----
            reward_hist = np.array(reward_hist)
            reward_hist = discount_rewards(reward_hist, adv)
            if lr > config.lr_rl * 0.1:
                lr = lr * config.lr_decay_rl

            # ----Randomly pick a fraction of steps to calc gradients,
            # repeat N times.----
            for n in range(config.max_ctrl_step):
                rand_inds = np.arange(len(state_hist))
                np.random.shuffle(rand_inds)
                rand_inds = np.sort(rand_inds[:int(len(state_hist) / 2)])
                sh = np.array(state_hist)[rand_inds]
                ah = np.array(action_hist)[rand_inds]
                rh = np.array(reward_hist)[rand_inds]

                grads = model_ctrl.get_gradients(sess_ctrl, sh, ah, rh, lr)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                logger.info('UPDATE CONTROLLOR')
                feed_dict = dict(zip(model_ctrl.gradient_plhs, gradBuffer))
                _ = sess_ctrl.run(model_ctrl.train_op, feed_dict=feed_dict)

                # ----Print gradients and weights.----
                logger.info('Gradients')
                for idx, grad in enumerate(gradBuffer):
                    logger.info(gradBuffer[idx])
                if np.isnan(gradBuffer[0][0][0]):
                    exit()
                logger.info('Weights')
                model_ctrl.print_weight(sess_ctrl)
                logger.info('Outputs')

                index = []
                ind = 3
                while ind < len(state_hist):
                    index.append(ind-3)
                    index.append(ind-2)
                    index.append(ind-1)
                    ind += 500
                feed_dict = {model_ctrl.state_plh:np.array(state_hist)[index],
                            model_ctrl.action_plh:np.array(action_hist)[index],
                            model_ctrl.reward_plh:np.array(reward_hist)[index]}
                fetch = [model_ctrl.output,
                            model_ctrl.action,
                            model_ctrl.indexes,
                            model_ctrl.responsible_outputs,
                            model_ctrl.reward_plh,
                            model_ctrl.state_plh]
                r = sess_ctrl.run(fetch, feed_dict=feed_dict)
                logger.info('state:\n{}'.format(r[5]))
                logger.info('output:\n{}'.format(r[0]))
                logger.info('action: {}'.format(r[1]))
                logger.info('indexes: {}'.format(r[2]))
                logger.info('res_outs: {}'.format(r[3]))
                logger.info('reward: {}'.format(r[4]))

                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # ----Study the relation between loss and action.----
                loss_analyzer(action_hist, valid_loss_hist, train_loss_hist,
                            reward_hist)


            #if final_reward > best_reward:
            #    best_reward = final_reward
            #    model_ctrl.save_model(sess_ctrl, global_step=ep)


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'regression.cfg'
    config_path = os.path.join(root_path, 'config/' + config_file)
    config = utils.Parser(config_path)
    trainer = Trainer(config)
    load_ctrl = os.path.join(config.model_dir, 'autoLoss-toy/')
    # ----start from pretrained----
    #trainer.train(load_ctrl=load_ctrl)
    # ----start from strach----
    trainer.train()
