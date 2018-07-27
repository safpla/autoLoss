""" Optimize mse loss, l1 loss, l2 loss at the same time """
# __Author__ == "Haowen Xu"
# __Data__ == "04-09-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys
from models import reg
import utils

logger = utils.get_logger()

def train(config):
    model = reg.Reg(config, loss_mode=sys.argv[1])
    model.initialize_weights()

    max_training_step = config.max_training_step
    best_loss = 1e10
    endurance = 0
    i = 0
    while i < max_training_step and endurance < config.max_endurance_stud:
        train_loss = model.train()
        if i % 10 == 0:
            endurance += 1
            valid_loss, _, _ = model.valid()
            if valid_loss < best_loss:
                best_loss = valid_loss
                with model.graph.as_default():
                    best_tvars = tf.trainable_variables()
                endurance = 0
        i += 1
    logger.info('lambda1: {}'.format(config.lambda1_stud))
    logger.info('best_loss: {}'.format(best_loss))
    return best_loss




if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/regression.cfg')
    config = utils.Parser(config_path)
    if sys.argv[1] == '1':
        #lambda_set1 = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        lambda_set1 = 0.3 + (np.array(range(21))) * 0.01
        num1 = len(lambda_set1)
        aver_mat = np.zeros([num1])
        var_mat = np.zeros([num1])
        mat = []
        for i in range(num1):
            config.lambda1_stud = lambda_set1[i]
            loss = []
            for k in range(1):
                loss.append(train(config))
            aver_mat[i] = np.mean(np.array(loss))
            var_mat[i] = np.var(np.array(loss))
            mat.append(loss)
        print(aver_mat)
        print(var_mat)
        print(mat)
    elif sys.argv[1] == '3':
        lambda_set1 = [0.2, 0.3, 0.4, 0.5, 0.6]
        lambda_set2 = [0.0001, 0.0003, 0.001, 0.003, 0.01]
        num1 = len(lambda_set1)
        num2 = len(lambda_set2)
        aver_mat = np.zeros([num1, num2])
        for i in range(num1):
            for j in range(num2):
                config.lambda1_stud = lambda_set1[i]
                config.lambda2_stud = lambda_set2[j]
                loss = []
                for k in range(1):
                    loss.append(train(config))
                aver_mat[i, j] = np.mean(np.array(loss))
        print(aver_mat)
    else:
        loss = []
        for k in range(1):
            loss.append(train(config))
        print(loss)
        print('\n')
        print(np.mean(np.array(loss)))
