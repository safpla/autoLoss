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
from models import gan
from models import gan_grid
import utils
from utils.analyse_utils import loss_analyzer_toy
from utils.analyse_utils import loss_analyzer_gan
import socket


root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

def discount_rewards(reward, final_reward):
    # TODO(haowen) Final reward + step reward
    reward_dis = np.array(reward) + np.array(final_reward)
    return reward_dis

class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config, exp_name='new_exp'):
        self.config = config
        self.model_ctrl = controller.Controller(config)
        if config.student_model_name == 'toy':
            self.model_stud = toy.Toy(config)
        elif config.student_model_name == 'cls':
            self.model_stud = cls.Cls(config)
        elif config.student_model_name == 'gan':
            self.model_stud = gan.Gan(config, exp_name=exp_name)
        elif config.student_model_name == 'gan_grid':
            self.model_stud = gan_grid.Gan_grid(config, exp_name=exp_name)
        else:
            raise NotImplementedError

    def train(self, save_ctrl=None, load_ctrl=None):
        """ Iteratively training between controller and the student model """
        config = self.config
        lr = config.lr_rl
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        best_reward = -1e5
        best_acc = 0
        best_loss = 0
        best_inps = 0
        best_best_inps = 0
        endurance = 0

        # ----Initialize controllor.----
        model_ctrl.initialize_weights()
        if load_ctrl:
            model_ctrl.load_model(load_ctrl)

        # ----Initialize gradient buffer.----
        gradBuffer = model_ctrl.get_weights()
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # ----Start episodes.----
        for ep in range(config.total_episodes):
            logger.info('=================')
            logger.info('episodes: {}'.format(ep))

            # ----Initialize student model.----
            model_stud.initialize_weights()
            if config.pretrained_gan_exp_name:
                pretrained_path = os.path.join(config.model_dir,
                                            config.pretrained_gan_exp_name)
                model_stud.load_model(pretrained_path)
            model_stud.reset()

            state = model_stud.get_state()
            state_hist = []
            action_hist = []
            reward_hist = []
            valid_loss_hist = []
            train_loss_hist = []
            old_action = []
            gen_cost_hist = []
            disc_cost_real_hist = []
            disc_cost_fake_hist = []
            step = -1
            # ----Running one episode.----
            while True:
                step += 1
                explore_rate = config.explore_rate_rl *\
                    math.exp(-ep / config.explore_rate_decay_rl)
                action = model_ctrl.sample(state, explore_rate=explore_rate)
                state_new, reward, dead = model_stud.response(action)
                # ----Record training details.----
                state_hist.append(state)
                action_hist.append(action)
                reward_hist.append(reward)
                if 'gan' in config.student_model_name:
                    gen_cost_hist.append(model_stud.ema_gen_cost)
                    disc_cost_real_hist.append(model_stud.ema_disc_cost_real)
                    disc_cost_fake_hist.append(model_stud.ema_disc_cost_fake)
                else:
                    valid_loss_hist.append(model_stud.previous_valid_loss[-1])
                    train_loss_hist.append(model_stud.previous_train_loss[-1])

                # ----Print training details.----
                #if step % 200 < 200:
                #    logger.info('----train_step: {}----'.format(step))
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
                #    model_stud.print_weights()

                old_action = action
                state = state_new
                if dead:
                    break

            # ----Update the controller.----
            final_reward, adv = model_stud.get_final_reward()
            reward_hist = np.array(reward_hist)
            reward_hist = discount_rewards(reward_hist, adv)
            sh = np.array(state_hist)
            ah = np.array(action_hist)
            rh = np.array(reward_hist)
            grads = model_ctrl.get_gradients(sh, ah, rh)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if lr > config.lr_rl * 0.1:
                lr = lr * config.lr_decay_rl
            if ep % config.update_frequency == (config.update_frequency - 1):
                logger.info('UPDATE CONTROLLOR')
                logger.info('lr_ctrl: {}'.format(lr))
                model_ctrl.train_one_step(gradBuffer, lr)
                logger.info('grad')
                for ix, grad in enumerate(gradBuffer):
                    logger.info(grad)
                    gradBuffer[ix] = grad * 0
                logger.info('weights')
                model_ctrl.print_weights()

                # ----Print training details.----
                logger.info('Outputs')
                index = []
                ind = 1
                while ind < len(state_hist):
                    index.append(ind-1)
                    ind += 2000
                feed_dict = {model_ctrl.state_plh:np.array(state_hist)[index],
                            model_ctrl.action_plh:np.array(action_hist)[index],
                            model_ctrl.reward_plh:np.array(reward_hist)[index]}
                fetch = [model_ctrl.output,
                         model_ctrl.action,
                         model_ctrl.reward_plh,
                         model_ctrl.state_plh,
                         model_ctrl.logits
                        ]
                r = model_ctrl.sess.run(fetch, feed_dict=feed_dict)
                logger.info('state:\n{}'.format(r[3]))
                logger.info('output:\n{}'.format(r[0]))
                logger.info('action: {}'.format(r[1]))
                logger.info('reward: {}'.format(r[2]))
                logger.info('logits: {}'.format(r[4]))

            save_model_flag = False
            if config.student_model_name == 'toy':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                  train_loss_hist, reward_hist)
                loss = model_stud.best_loss
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_loss = loss
                    save_model_flag = Ture
                logger.info('best_loss: {}'.format(loss))
                logger.info('lambda1: {}'.format(config.lambda1_stud))
            elif config.student_model_name == 'cls':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                  train_loss_hist, reward_hist)
                acc = model_stud.best_acc
                loss = model_stud.best_loss
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_acc = acc
                    best_loss = loss
                logger.info('acc: {}'.format(acc))
                logger.info('best_acc: {}'.format(best_acc))
                logger.info('best_loss: {}'.format(loss))
                logger.info('lambda1: {}'.format(config.lambda1_stud))
                if ep % cofig.save_frequency == 0 and ep > 0:
                    save_model_flag = True
            elif config.student_model_name == 'gan':
                loss_analyzer_gan(action_hist, reward_hist)
                best_inps = model_stud.best_inception_score
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_best_inps = best_inps
                    save_model_flag = True
                logger.info('best_inps: {}'.format(best_inps))
                logger.info('best_best_inps: {}'.format(best_best_inps))
            elif config.student_model_name == 'gan_grid':
                loss_analyzer_gan(action_hist, reward_hist)
                hq_ratio = model_stud.best_hq_ratio
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_hq_ratio = hq_ratio
                    save_model_flag = True
                logger.info('hq_ratio: {}'.format(hq_ratio))
                logger.info('best_hq_ratio: {}'.format(best_hq_ratio))

            logger.info('final_reward: {}'.format(final_reward))
            logger.info('adv: {}'.format(adv))

            if save_model_flag and save_ctrl:
                model_ctrl.save_model(ep)

    def test(self, load_ctrl, ckpt_num=None):
        config = self.config
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        model_ctrl.initialize_weights()
        model_stud.initialize_weights()
        model_ctrl.load_model(load_ctrl, ckpt_num=ckpt_num)
        model_stud.reset()

        state = model_stud.get_state()
        for i in range(config.max_training_step):
            action = model_ctrl.sample(state)
            state_new, _, dead = model_stud.response(action)
            state = state_new
            if dead:
                break
        valid_acc = model_stud.best_acc
        test_acc = model_stud.test_acc
        logger.info('valid_acc: {}'.format(valid_acc))
        logger.info('test_acc: {}'.format(test_acc))
        return test_acc



if __name__ == '__main__':
    # ----Parsing config file.----
    logger.info(socket.gethostname())
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'gan.cfg'
    config_path = os.path.join(root_path, 'config/' + config_file)
    config = utils.Parser(config_path)

    # ----Instantiate a trainer object.----
    trainer = Trainer(config, exp_name='dcgan_mnist')

    # ----Training----
    #   --Start from pretrained--
    #trainer.train(load_ctrl=load_ctrl)
    #   --Start from strach--
    model_path = os.path.join(config.model_dir, '')
    trainer.train(save_ctrl=model_path)
    #trainer.train()

    # ----Testing----
    #test_accs = []
    ##ckpt_num = 250
    #ckpt_num = None
    #for i in range(10):
    #    test_accs.append(trainer.test(model_path, ckpt_num=ckpt_num))
    #print(test_accs)
    #print(np.mean(np.array(test_accs)))
