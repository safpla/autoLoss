""" Optimize mse loss, l1 loss, l2 loss at the same time """
# __Author__ == "Haowen Xu"
# __Data__ == "04-09-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys
from models import toy
import utils

logger = utils.get_logger()

def train(config):
    g = tf.Graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    configProto = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=configProto, graph=g)

    model = toy.Toy(config, g, sys.argv[1])
    sess.run(model.init)

    max_training_step = config.max_training_step
    best_loss = 1e10
    endurance = 0
    i = 0
    while i < max_training_step and endurance < 100:
        train_loss = model.train(sess)
        if i % 10 == 0:
            endurance += 1
            valid_loss, _, _ = model.valid(sess)
            logger.info('step: {}, train_loss: {}, valid_loss: {}'\
                        .format(i, train_loss, valid_loss))
            if valid_loss < best_loss:
                best_loss = valid_loss
                endurance = 0
        i += 1

    ## print weights
    #with model.graph.as_default():
    #    tvars = tf.trainable_variables()
    #    for tvar in tvars:
    #        print(sess.run(tvar))

    ## print results on validation set
    #valid_loss, preds, gdths = model.valid(sess)
    #for pred, gdth in zip(preds, gdths):
    #    print('pred: {} ---- gdth: {}'.format(pred, gdth))

    logger.info('best_loss: {}'.format(best_loss))
    return best_loss




if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/regression.cfg')
    config = utils.Parser(config_path)
    if sys.argv[1] == 2:
        lambda_set = [0.0001, 0.0002, 0.0005, 0.001, 0.002]
        aver_mat = np.zeros([5,5])
        for i in range(5):
            for j in range(5):
                config.lambda1_stud = lambda_set[i]
                config.lambda2_stud = lambda_set[j]
                loss = []
                for k in range(10):
                    loss.append(train(config))
                aver_mat[i, j] = np.mean(np.array(loss))
        print(aver_mat)
    else:
        loss = []
        for k in range(10):
            loss.append(train(config))
        print(np.mean(np.array(loss)))
