""" The module for training autoLoss """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys
import math
import socket
from time import gmtime, strftime

from models import controller
from models import reg
from models import cls
from models import gan
from models import gan_grid
from models import gan_cifar10
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
    def __init__(self, config, exp_name=None, arch=None):
        self.config = config

        hostname = socket.gethostname()
        hostname = '-'.join(hostname.split('.')[0:2])
        datetime = strftime('%m-%d-%H-%M', gmtime())
        if not exp_name:
            exp_name = '{}_{}'.format(hostname, datetime)
        logger.info('exp_name: {}'.format(exp_name))

        self.model_ctrl = controller.Controller(config, exp_name+'_ctrl')
        if config.student_model_name == 'reg':
            self.model_stud = reg.Reg(config, exp_name+'_reg')
        elif config.student_model_name == 'cls':
            self.model_stud = cls.Cls(config, exp_name+'_cls')
        elif config.student_model_name == 'gan':
            self.model_stud = gan.Gan(config, exp_name+'_gan', arch=arch)
        elif config.student_model_name == 'gan_grid':
            self.model_stud = gan_grid.Gan_grid(config, exp_name+'_gan_grid')
        elif config.student_model_name == 'gan_cifar10':
            self.model_stud = gan_cifar10.Gan_cifar10(config,
                                                      exp_name+'_gan_cifar10')
        else:
            raise NotImplementedError

    def train(self, save_ctrl=None, load_ctrl=None):
        """ Iteratively training between controller and the student model """
        config = self.config
        lr = config.lr_rl
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        best_reward = -1e5 # A big negative initial value
        best_acc = 0
        best_loss = 0
        best_inps = 0
        best_best_inps = 0
        endurance = 0
        lrs = np.linspace(config.lr_start_stud,
                          config.lr_end_stud,
                          config.lr_decay_steps_stud)

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
                lr = lrs[min(config.lr_decay_steps_stud-1, step)]
                action = model_ctrl.sample(state, explore_rate=explore_rate)
                state_new, reward, dead = model_stud.response(action, lr)
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
                #if step > 0:
                #    logger.info('----train_step: {}----'.format(step))
                #    logger.info('state:{}'.format(state_new))
                #    logger.info('action: {}'.format(action))
                #    logger.info('reward:{}'.format(reward))
                #    #lv = model_stud.previous_valid_loss
                #    #lt = model_stud.previous_train_loss
                #    #av = model_stud.previous_valid_acc
                #    #at = model_stud.previous_train_acc
                #    #logger.info('train_loss: {}'.format(lt[-1]))
                #    #logger.info('valid_loss: {}'.format(lv[-1]))
                #    #logger.info('loss_imp: {}'.format(lv[-2] - lv[-1]))
                #    #logger.info('train_acc: {}'.format(at[-1]))
                #    #logger.info('valid_acc: {}'.format(av[-1]))
                #    #model_stud.print_weights()

                old_action = action
                state = state_new
                if dead:
                    break

            # ----Use the best performance model to get inception score on a
            #     larger number of samples to reduce the variance of reward----
            if config.student_model_name == 'gan' and model_stud.task_dir:
                model_stud.load_model(model_stud.task_dir)
                inps_test = model_stud.get_inception_score(5000)
                logger.info('inps_test: {}'.format(inps_test))
                model_stud.update_inception_score(inps_test[0])
            else:
                logger.info('student model hasn\'t been saved before')


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
                #logger.info('grad')
                #for ix, grad in enumerate(gradBuffer):
                #    logger.info(grad)
                #    gradBuffer[ix] = grad * 0
                #logger.info('weights')
                #model_ctrl.print_weights()

                # ----Print training details.----
                #logger.info('Outputs')
                #index = []
                #ind = 1
                #while ind < len(state_hist):
                #    index.append(ind-1)
                #    ind += 2000
                #feed_dict = {model_ctrl.state_plh:np.array(state_hist)[index],
                #            model_ctrl.action_plh:np.array(action_hist)[index],
                #            model_ctrl.reward_plh:np.array(reward_hist)[index]}
                #fetch = [model_ctrl.output,
                #         model_ctrl.action,
                #         model_ctrl.reward_plh,
                #         model_ctrl.state_plh,
                #         model_ctrl.logits
                #        ]
                #r = model_ctrl.sess.run(fetch, feed_dict=feed_dict)
                #logger.info('state:\n{}'.format(r[3]))
                #logger.info('output:\n{}'.format(r[0]))
                #logger.info('action: {}'.format(r[1]))
                #logger.info('reward: {}'.format(r[2]))
                #logger.info('logits: {}'.format(r[4]))

            save_model_flag = False
            if config.student_model_name == 'reg':
                loss_analyzer_toy(action_hist, valid_loss_hist,
                                  train_loss_hist, reward_hist)
                loss = model_stud.best_loss
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_loss = loss
                    save_model_flag = True
                    endurance = 0
                else:
                    endurance += 1
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
                    save_model_flag = True
                    enduranc = 0
                else:
                    endurance += 1
                logger.info('acc: {}'.format(acc))
                logger.info('best_acc: {}'.format(best_acc))
                logger.info('best_loss: {}'.format(loss))
                #if ep % config.save_frequency == 0 and ep > 0:
                #    save_model_flag = True
            elif config.student_model_name == 'gan' or\
                config.student_model_name == 'gan_cifar10':
                loss_analyzer_gan(action_hist, reward_hist)
                best_inps = model_stud.best_inception_score
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_best_inps = best_inps
                    save_model_flag = True
                logger.info('best_inps: {}'.format(best_inps))
                logger.info('best_best_inps: {}'.format(best_best_inps))
                logger.info('final_inps_baseline: {}'.\
                            format(model_stud.final_inps_baseline))
            elif config.student_model_name == 'gan_grid':
                loss_analyzer_gan(action_hist, reward_hist)
                hq = model_stud.best_hq
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_hq = hq
                    save_model_flag = True
                logger.info('hq: {}'.format(hq))
                logger.info('best_hq: {}'.format(best_hq))

            logger.info('adv: {}'.format(adv))

            if save_model_flag and save_ctrl:
                    model_ctrl.save_model(ep)
            #if endurance > config.max_endurance_rl:
            #    break

    def test(self, load_ctrl, ckpt_num=None):
        config = self.config
        model_ctrl = self.model_ctrl
        model_stud = self.model_stud
        model_ctrl.initialize_weights()
        model_stud.initialize_weights()
        model_ctrl.load_model(load_ctrl, ckpt_num=ckpt_num)
        model_stud.reset()
        lrs = np.linspace(config.lr_start_stud,
                          config.lr_end_stud,
                          config.lr_decay_steps_stud)

        state = model_stud.get_state()
        for i in range(config.max_training_step):
            lr = lrs[min(config.lr_decay_steps_stud-1, i)]
            action = model_ctrl.sample(state)
            state_new, _, dead = model_stud.response(action, lr, mode='TEST')
            if (i % 10 == 0) and config.student_model_name == 'cls':
                valid_loss = model_stud.previous_valid_loss[-1]
                valid_acc = model_stud.previous_valid_acc[-1]
                train_loss = model_stud.previous_train_loss[-1]
                train_acc = model_stud.previous_train_acc[-1]
                logger.info('Step {}'.format(i))
                logger.info('train_loss: {}, valid_loss: {}'.format(train_loss, valid_loss))
                logger.info('train_acc : {}, valid_acc : {}'.format(train_acc, valid_acc))

            state = state_new
            if dead:
                break
        if config.student_model_name == 'reg':
            loss = model_stud.best_loss
            logger.info('loss: {}'.format(loss))
            return loss
        elif config.student_model_name == 'cls':
            valid_acc = model_stud.best_acc
            test_acc = model_stud.test_acc
            logger.info('valid_acc: {}'.format(valid_acc))
            logger.info('test_acc: {}'.format(test_acc))
            return test_acc
        elif config.student_model_name == 'gan':
            model_stud.load_model(model_stud.task_dir)
            inps_test = model_stud.get_inception_score(500, splits=5)
            model_stud.genrate_images(0)
            logger.info('inps_test: {}'.format(inps_test))
            return inps_test
        elif config.student_model_name == 'gan_grid':
            raise NotImplementedError
        elif config.student_model_name == 'gan_cifar10':
            best_inps = model_stud.best_inception_score
            logger.info('best_inps: {}'.format(best_inps))
            return best_inps
        else:
            raise NotImplementedError

    def baseline(self):
        self.model_stud.initialize_weights()
        self.model_stud.train(save_model=True)
        if self.config.student_model_name == 'gan':
            self.model_stud.load_model(self.model_stud.task_dir)
            inps_baseline = self.model_stud.get_inception_score(500, splits=5)
            self.model_stud.generate_images(0)
            logger.info('inps_baseline: {}'.format(inps_baseline))
        return inps_baseline

    def generate(self, load_stud):
        self.model_stud.initialize_weights()
        self.model_stud.load_model(load_stud)
        self.model_stud.generate_images(0)


if __name__ == '__main__':
    argv = sys.argv
    # ----Parsing config file.----
    logger.info(socket.gethostname())
    #config_file = 'gan.cfg'
    config_file = 'regression.cfg'
    #config_file = 'gan_cifar10.cfg'
    #config_file = 'classification.cfg'
    config_path = os.path.join(root_path, 'config/' + config_file)
    config = utils.Parser(config_path)
    config.print_config()

    # ----Instantiate a trainer object.----
    trainer = Trainer(config, exp_name=argv[1])

    # classification task controllor model
    #load_ctrl = '/datasets/BigLearning/haowen/autoLoss/saved_models/h5-haowen6_05-13-04-30_ctrl'
    # regression task controllor model
    #load_ctrl = '/datasets/BigLearning/haowen/autoLoss/saved_models/h2-haowen6_05-15-20-32_ctrl'
    # gan mnist task controllor model
    #load_ctrl = '/datasets/BigLearning/haowen/autoLoss/saved_models/h2-haowen6_05-13-20-21_ctrl'

    load_stud = '/datasets/BigLearning/haowen/autoLoss/saved_models/' + argv[1]

    if argv[2] == 'train':
        # ----Training----
        #   --Start from pretrained--
        #trainer.train(load_ctrl=load_ctrl)
        #trainer.train(load_ctrl=load_ctrl, save_ctrl=True)
        #   --Start from strach--
        trainer.train(save_ctrl=True)
        #trainer.train(save_ctrl=False)
    elif argv[2] == 'test':
        ## ----Testing----
        logger.info('TEST')
        test_accs = []
        for i in range(1):
            test_accs.append(trainer.test(trainer.model_ctrl.task_dir))
            #test_accs.append(trainer.test(load_ctrl))
        logger.info(test_accs)
    elif argv[2] == 'baseline':
        # ----Baseline----
        logger.info('BASELINE')
        baseline_accs = []
        for i in range(1):
            baseline_accs.append(trainer.baseline())
        logger.info(baseline_accs)
    elif argv[2] == 'generate':
        logger.info('GENERATE')
        trainer.generate(load_stud)


