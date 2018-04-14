""" Generate toy dataset for logistic regression with L1, L2 regularization """

# __author__ == 'Haowen Xu'
# __data__ == '04_07_2018'

import os, sys
import numpy as np
import json

dim = 32
num_of_data_points_train = 500
num_of_data_points_valid = 1000
mean_noise = 0
var_noise = 2
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_train = os.path.join(father_dir, 'Data/toy/train.npy')
data_valid = os.path.join(father_dir, 'Data/toy/valid.npy')
np.random.seed(1652)


def linear_func(x, w):
    return np.dot(x, w)

def main():
    # generator training dataset
    w = np.random.rand(dim) - 0.5
    #w = np.array([0.1, 0.5, 0.5, -0.1, 0.2, -0.2, 0.3, 0.1])
    print('weight: ', w)

    with open(data_train, 'wb') as f:
        data = []
        for i in range(num_of_data_points_train):
            x = np.random.rand(dim) - 0.5
            y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                     scale=var_noise)
            data.append({'x': x, 'y': y})

        np.save(f, data)
        f.close()

    with open(data_valid, 'wb') as f:
        data = []
        for i in range(num_of_data_points_valid):
            x = np.random.rand(dim)
            y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                     scale=var_noise)
            data.append({'x': x, 'y': y})

        np.save(f, data)
        f.close()

    mse = 0
    for d in data:
        x = d['x']
        y = d['y']
        y_ = linear_func(x, w)
        mse += np.square(y - y_)
    mse /= num_of_data_points_valid
    print('valid_loss:', mse)

def load():
    with open(data_train, 'rb') as f:
        data = np.load(f)
        print(data)


if __name__ == '__main__':
    main()
