import numpy as np
import utils

logger = utils.get_logger()


def loss_analyzer(actions, valid_losses, train_losses):
    total_steps = len(actions)
    # ----Prior of each action.----
    action_sum = np.sum(np.array(actions), axis=0) / total_steps
    logger.info('p_a: {}'.format(action_sum))

    loss_mse = []
    for idx, a in enumerate(actions):
        if a[0] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_mse.append(loss_diff)

    loss_l1 = []
    for idx, a in enumerate(actions):
        if a[1] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_l1.append(loss_diff)

    loss_l2 = []
    for idx, a in enumerate(actions):
        if a[2] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_l2.append(loss_diff)

    # ----Mean and Var of loss improvement of each action.----
    logger.info('loss_mse_mean: {}'.format(np.mean(np.array(loss_mse))))
    logger.info('loss_mse_var: {}'.format(np.var(np.array(loss_mse))))
    logger.info('loss_l1_mean: {}'.format(np.mean(np.array(loss_l1))))
    logger.info('loss_l1_var: {}'.format(np.var(np.array(loss_l1))))
    logger.info('loss_l2_mean: {}'.format(np.mean(np.array(loss_l2))))
    logger.info('loss_l2_var: {}'.format(np.var(np.array(loss_l2))))

    # ----Distribution of each action over time.----

