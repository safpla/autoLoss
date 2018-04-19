import numpy as np
import utils

logger = utils.get_logger()

def get_reward(losses):
    zero = 0
    pos = 0
    neg = 0
    for l in losses:
        if abs(l) < 1e-5:
            zero += 1
        elif l > 0:
            pos += 1
        else:
            neg += 1
    return zero, pos, neg

def loss_analyzer(actions, valid_losses, train_losses):
    total_steps = len(actions)
    logger.info('total_steps: {}'.format(total_steps))

    # ----Prior of each action.----
    action_sum = np.sum(np.array(actions), axis=0) / total_steps
    logger.info('p_a: {}'.format(action_sum))

    loss_mse = []
    for idx, a in enumerate(actions):
        if idx == 0:
            continue
        if a[0] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_mse.append(loss_diff)

    loss_l1 = []
    for idx, a in enumerate(actions):
        if idx == 0:
            continue
        if a[1] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_l1.append(loss_diff)

    loss_l2 = []
    for idx, a in enumerate(actions):
        if idx == 0:
            continue
        if a[2] == 1:
            loss_diff = valid_losses[idx - 1] - valid_losses[idx]
            loss_l2.append(loss_diff)

    # ----Mean and Var of loss improvement of each action.----
    logger.info('loss_mse_mean: {}, var: {}'.format(
        np.mean(np.array(loss_mse)), np.var(np.array(loss_mse))))
    logger.info('loss_l1_mean: {}, var: {}'.format(
        np.mean(np.array(loss_l1)), np.var(np.array(loss_l1))))
    logger.info('loss_l2_mean: {}, var: {}'.format(
        np.mean(np.array(loss_l2)), np.var(np.array(loss_l2))))

    # ----Step reward distribution.----
    zero, pos, neg = get_reward(loss_mse)
    logger.info('MSE:: zero: {}, pos: {}, neg: {}'.format(zero, pos, neg))

    zero, pos, neg = get_reward(loss_l1)
    logger.info('L1 :: zero: {}, pos: {}, neg: {}'.format(zero, pos, neg))

    zero, pos, neg = get_reward(loss_l2)
    logger.info('L2 :: zero: {}, pos: {}, neg: {}'.format(zero, pos, neg))
    # ----Distribution of each action over time.----
    win = 100
    loss_imp_mse_trace = []
    loss_imp_l1_trace = []
    loss_imp_l2_trace = []
    mse_dis_trace = []
    l1_dis_trace = []
    l2_dis_trace = []
    for i in range(min(80, int(total_steps / win))):
        start = i * win
        stop = (i + 1) * win
        action = actions[start:stop]
        valid_loss = valid_losses[start:stop]
        loss_mse = []
        for idx, a in enumerate(action):
            if idx == 0:
                continue
            if a[0] == 1:
                loss_diff = valid_loss[idx - 1] - valid_loss[idx]
                loss_mse.append(loss_diff)

        loss_l1 = []
        for idx, a in enumerate(action):
            if idx == 0:
                continue
            if a[1] == 1:
                loss_diff = valid_loss[idx - 1] - valid_loss[idx]
                loss_l1.append(loss_diff)

        loss_l2 = []
        for idx, a in enumerate(action):
            if idx == 0:
                continue
            if a[2] == 1:
                loss_diff = valid_loss[idx - 1] - valid_loss[idx]
                loss_l2.append(loss_diff)
        loss_imp_mse_trace.append(np.mean(np.array(loss_mse)))
        loss_imp_l1_trace.append(np.mean(np.array(loss_l1)))
        loss_imp_l2_trace.append(np.mean(np.array(loss_l2)))
        mse_dis_trace.append(len(loss_mse))
        l1_dis_trace.append(len(loss_l1))
        l2_dis_trace.append(len(loss_l2))

    logger.info('Trace of actions distribution')
    logger.info('mse: {}'.format(mse_dis_trace))
    logger.info('l1: {}'.format(l1_dis_trace))
    logger.info('l2: {}'.format(l2_dis_trace))

    logger.info('Trace of loss improvement:')
    logger.info('mse: {}'.format(loss_imp_mse_trace))
    logger.info('l1: {}'.format(loss_imp_l1_trace))
    logger.info('l2: {}'.format(loss_imp_l2_trace))
