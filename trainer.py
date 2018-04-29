""" The module for training autoLoss """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys
import math

from models import controller
from models import toy
from models import cls
import utils
from utils.analyse_utils import loss_analyzer_toy


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
        self.g_stud = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess_ctrl = tf.InteractiveSession(config=configProto,
                                               graph=self.g_ctrl)
        self.model_ctrl = controller.Controller(config, self.g_ctrl)
        if config.student_model_name == 'toy':
            self.model_stud = toy.Toy(config, self.g_stud)
        elif config.student_model_name == 'cls':
            self.model_stud = cls.Cls(config, self.g_stud)
        else:
            raise NotImplementedError

    def train(self, save_ctrl=None, load_ctrl=None):
        """ Iteratively training between controller and the multi-loss task """
        config = self.config
        sess_ctrl = self.sess_ctrl
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        best_reward = -1e5
        best_acc = 0
        best_loss = 0
        endurance = 0
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
            state_hist = []
            action_hist = []
            reward_hist = []
            valid_loss_hist = []
            train_loss_hist = []
            old_action = []

            # ----Running one episode.----
            for i in range(config.max_training_step):
                explore_rate = config.explore_rate_rl *\
                    math.exp(-ep / config.explore_rate_decay_rl)
                action = model_ctrl.sample(sess_ctrl, state,
                                           explore_rate=explore_rate)
                state_new, reward, dead = model_stud.env(sess_stud, action)
                state_hist.append(state)
                action_hist.append(action)
                reward_hist.append(reward)
                valid_loss_hist.append(model_stud.previous_valid_loss[-1])
                train_loss_hist.append(model_stud.previous_train_loss[-1])

                #if i % 1 == 0 and i < 50:
                #    logger.info('----train_step: {}----'.format(i))
                #    logger.info('state:{}'.format(state_new))
                #    logger.info('action: {}'.format(action))
                #    logger.info('reward:{}'.format(reward))
                #    lv = model_stud.previous_valid_loss
                #    lt = model_stud.previous_train_loss
                #    av = model_stud.previous_valid_acc
                #    at = model_stud.previous_train_acc
                #    logger.info('train_loss: {}'.format(lt[-1]))
                #    logger.info('valid_loss: {}'.format(lv[-1]))
                #    logger.info('loss_imp: {}'.format(lv[-2] - lv[-1]))
                #    logger.info('train_acc: {}'.format(at[-1]))
                #    logger.info('valid_acc: {}'.format(av[-1]))
                #    model_stud.print_weight(sess_stud)

                old_action = action
                state = state_new
                if dead:
                    break

            # ----Only use the history before the best result.----
            # ----(Deserted)----
            #state_hist = state_hist[:model_stud.best_step]
            #action_hist = action_hist[:model_stud.best_step]
            #reward_hist = reward_hist[:model_stud.best_step]
            #valid_loss_hist = valid_loss_hist[:model_stud.best_step]
            #train_loss_hist = train_loss_hist[:model_stud.best_step]

            final_reward, adv = model_stud.get_final_reward()
            loss = model_stud.best_loss

            # ----Update the controller.----
            reward_hist = np.array(reward_hist)
            reward_hist = discount_rewards(reward_hist, adv)
            if lr > config.lr_rl * 0.1:
                lr = lr * config.lr_decay_rl
            sh = np.array(state_hist)
            ah = np.array(action_hist)
            rh = np.array(reward_hist)
            grads = model_ctrl.get_gradients(sess_ctrl, sh, ah, rh)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if ep % config.update_frequency == 0 and ep > 0:
                logger.info('UPDATE CONTROLLOR')
                feed_dict = dict(zip(model_ctrl.gradient_plhs, gradBuffer))
                feed_dict[model_ctrl.lr_plh] = lr
                logger.info('lr_ctrl: {}'.format(lr))
                _ = sess_ctrl.run(model_ctrl.train_op, feed_dict=feed_dict)

                # ----Print gradients and weights.----
                #logger.info('Gradients')
                #for idx, grad in enumerate(gradBuffer):
                #    logger.info(gradBuffer[idx])
                #if np.isnan(gradBuffer[0][0][0]):
                #    exit()
                #logger.info('Weights')
                #model_ctrl.print_weight(sess_ctrl)

                logger.info('Outputs')
                index = []
                ind = 1
                while ind < len(state_hist):
                    #index.append(ind-5)
                    #index.append(ind-4)
                    #index.append(ind-3)
                    #index.append(ind-2)
                    index.append(ind-1)
                    ind += 100
                feed_dict = {model_ctrl.state_plh:np.array(state_hist)[index],
                            model_ctrl.action_plh:np.array(action_hist)[index],
                            model_ctrl.reward_plh:np.array(reward_hist)[index]}
                fetch = [model_ctrl.output,
                         model_ctrl.action,
                         model_ctrl.reward_plh,
                         model_ctrl.state_plh,
                        ]
                r = sess_ctrl.run(fetch, feed_dict=feed_dict)
                logger.info('state:\n{}'.format(r[3]))
                logger.info('output:\n{}'.format(r[0]))
                logger.info('action: {}'.format(r[1]))
                logger.info('reward: {}'.format(r[2]))

                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

            # ----Study the relation between loss and action.----
            if config.student_model_name == 'toy':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                    train_loss_hist, reward_hist)
            elif config.student_model_name == 'cls':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                    train_loss_hist, reward_hist)
            logger.info('final_reward: {}'.format(final_reward))
            logger.info('loss: {}'.format(loss))
            if config.student_model_name == 'cls':
                acc = model_stud.best_acc
                logger.info('acc: {}'.format(acc))
            logger.info('adv: {}'.format(adv))
            logger.info('lambda1: {}'.format(config.lambda1_stud))
            logger.info('best_acc: {}'.format(best_acc))

            if ep % config.save_frequency == 0 and ep > 0:
                model_ctrl.save_model(sess_ctrl, save_ctrl, global_step=ep)

            if final_reward > best_reward:
                best_reward = final_reward
                if config.student_model_name == 'cls':
                    best_acc = acc
                best_loss = loss
                #if save_ctrl:
                #    model_ctrl.save_model(sess_ctrl, save_ctrl, global_step=ep)
                #endurance = 0
            #else:
            #    endurance += 1
            #if endurance > config.max_endurance_rl:
            #    if config.student_model_name == 'cls':
            #        logger.info('best_acc: {}'.format(best_acc))
            #    logger.info('best_loss: {}'.format(best_loss))
            #    break

    def test(self, load_ctrl, ckpt_num=None):
        config = self.config
        sess_ctrl = self.sess_ctrl
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        sess_stud = tf.InteractiveSession(config=configProto,
                                          graph=self.g_stud)

        sess_ctrl.run(model_ctrl.init)
        sess_stud.run(model_stud.init)
        model_ctrl.load_model(sess_ctrl, load_ctrl, ckpt_num=ckpt_num)
        model_stud.reset()

        state = model_stud.get_state(sess_stud)
        for i in range(config.max_training_step):
            action = model_ctrl.sample(sess_ctrl, state)
            state_new, _, dead = model_stud.env(sess_stud, action)
            state = state_new
            if dead:
                break
        valid_acc = model_stud.best_acc
        test_acc = model_stud.test_acc
        logger.info('valid_acc: {}'.format(valid_acc))
        logger.info('test_acc: {}'.format(test_acc))
        return test_acc



if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'classification.cfg'
    config_path = os.path.join(root_path, 'config/' + config_file)
    config = utils.Parser(config_path)
    trainer = Trainer(config)
    model_path = os.path.join(config.model_dir, 'autoLoss-cls-1l7s-{}/'.format(
        config.lambda1_stud))
    # ----start from pretrained----
    #trainer.train(load_ctrl=load_ctrl)
    # ----start from strach----
    trainer.train(save_ctrl=model_path)
    test_accs = []
    #ckpt_num = 250
    ckpt_num = None
    for i in range(10):
        test_accs.append(trainer.test(model_path, ckpt_num=ckpt_num))
    print(test_accs)
    print(np.mean(np.array(test_accs)))
