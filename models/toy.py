""" This module implement a toy task: linear regression """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Toy():
    def __init__(self, config, graph):
        self.config = config
        self.graph = graph

    def get_initialized_state(self):
        """ Return an initial state """
        # TODO(haowen) the way to carry step number information should be
        # reconsiderd
        step_number = [0]
        previous_mse_loss = [0] * self.config.num_pre_loss
        previous_l1_loss = [0] * self.config.num_pre_loss
        previous_l2_loss = [0] * self.config.num_pre_loss
        previous_action = [1, 0, 0]
        # TODO(haowen) simply concatenate them will cause scale problem
        state = step_number + previous_mse_loss + previous_l1_loss\
            + previous_l2_loss + previous_action
        return np.array(state, dtype='f')

    def _build_placeholder(self):
        pass

    def _inference_graph(self):
        pass

    def env(self, action):
        """ Given an action, return the new state, reward and whether dead

        Args:
            action: one hot encoding of actions

        Returns:
            state: shape = [dim_state_rl]
            reward: shape = [1]
            dead: boolean
        """
        pass
