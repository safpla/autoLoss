""" Optimize mse loss, l1 loss, l2 loss at the same time """
# __Author__ == "Haowen Xu"
# __Data__ == "04-25-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys
from models import cls
import utils

logger = utils.get_logger()

def train(config):
    model = cls.Cls(config, loss_mode=sys.argv[1])
    model.initialize_weights()

    max_training_step = config.max_training_step
    best_acc = 0
    endurance = 0
    i = 0
    while i < max_training_step and endurance < config.max_endurance_stud:
        train_loss, train_acc = model.train()
        if i % config.valid_frequency_stud == 0:
            endurance += 1
            valid_loss, valid_acc, _, _ = model.valid()
            #logger.info('====Step: {}===='.format(i))
            #logger.info('train_loss: {}, train_acc: {}'\
            #            .format(train_loss, train_acc))
            #logger.info('valid_loss: {}, valid_acc: {}'\
            #            .format(valid_loss, valid_acc))
            if valid_acc > best_acc:
                best_acc = valid_acc
                _, test_acc, _, _ = model.valid(model.test_dataset)
                endurance = 0
        i += 1

    logger.info('lambda1: {}, lambda2: {}'.format(config.lambda1_stud,
                                                  config.lambda2_stud))
    logger.info('valid_acc: {}'.format(best_acc))
    logger.info('test_acc: {}'.format(test_acc))
    return test_acc


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/classification_transfer.cfg')
    config = utils.Parser(config_path)
    if sys.argv[1] == '1':
        #lambda_set1 = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
        lambda_set1 = [0.03]
        num1 = len(lambda_set1)
        aver_mat = np.zeros([num1])
        mat = []
        for i in range(num1):
            config.lambda1_stud = lambda_set1[i]
            acc = []
            for k in range(5):
                acc.append(train(config))
            aver_mat[i] = np.mean(np.array(acc))
            mat.append(acc)
        print(aver_mat)
        print(mat)
    elif sys.argv[1] == '3':
        #lambda_set1 = [0.0001, 0.0003, 0.001, 0.003, 0.01]
        lambda_set1 = [0.02, 0.03, 0.04]
        lambda_set2 = [0.0001, 0.0003, 0.001, 0.003, 0.01]
        #lambda_set1 = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
        #lambda_set1 = [0.005, 0.006, 0.007, 0.008]
        num1 = len(lambda_set1)
        num2 = len(lambda_set2)
        aver_mat = np.zeros([num1, num2])
        var_mat = np.zeros([num1, num2])
        for i in range(num1):
            for j in range(num2):
                config.lambda1_stud = lambda_set1[i]
                config.lambda2_stud = lambda_set2[j]
                acc = []
                for k in range(5):
                    acc.append(train(config))
                aver_mat[i, j] = np.mean(np.array(acc))
                var_mat[i,j] = np.var(np.array(acc))
        print(aver_mat)
        print(var_mat)
    else:
        acc = []
        for k in range(5):
            acc.append(train(config))
        print(acc)
        print('\n')
        print(np.mean(np.array(acc)))
