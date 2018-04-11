""" Optimize mse loss, l1 loss, l2 loss at the same time """
# __Author__ == "Haowen Xu"
# __Data__ == "04-09-2018"

import tensorflow as tf
import numpy as np
import logging
import os

from models import toy
import utils

logger = utils.get_logger()

def train():
    g = tf.Graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    configProto = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=configProto, graph=g)

    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/regression.cfg')
    config = utils.Parser(config_path)

    model = toy.Toy(config, g)
    sess.run(model.init)

    max_training_step = config.max_training_step
    best_loss = 1e10
    best_reward = 0
    endurance = 0
    i = 0
    while i < max_training_step and endurance < 100:
        train_loss = model.train(sess)
        if i % 10 == 0:
            endurance += 1
            valid_loss, _, _ = model.valid(sess)
            reward = model.get_final_reward(sess)
            logger.info('step: {}, train_loss: {}, valid_loss: {}, reward: {}'\
                        .format(i, train_loss, valid_loss, reward))
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_reward = reward
                endurance = 0
        i += 1

    ## print weights
    #with model.graph.as_default():
    #    tvars = tf.trainable_variables()
    #    for tvar in tvars:
    #        print(sess.run(tvar))

    # print results on validation set
    valid_loss, preds, gdths = model.valid(sess)
    for pred, gdth in zip(preds, gdths):
        print('pred: {} ---- gdth: {}'.format(pred, gdth))

    logger.info('best_loss: {}, best_reward: {}'.format(best_loss, best_reward))




if __name__ == '__main__':
    train()
