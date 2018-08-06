import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import numpy.matlib
import matplotlib as mpl
import os
import sys

import log_utils
name = ''
engine = ''

def lineplot(data, colors, labels, fig_name):
    #x1 = np.array(range(len(data[0])))
    #x2 = np.array(range(len(data[1])))
    #y1 = np.array(data[0])
    #y2 = np.array(data[1])
    num = len(data)
    x = []
    y = []
    for d in data:
        x.append(np.array(range(len(d))))
        y.append(np.array(d))


    # Plot code
    markersize = 9
    ticksize = 14
    linewidth = 1.5
    legendfont = 14

    labelfont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 19}

    titlefont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 42,
            'weight' : 'bold'}

    fig, ax = plt.subplots()
    for i in range(num):
        ax.plot(x[i], y[i], color=colors[i],
                linewidth=linewidth, label=labels[i])
    #ax.plot(x1, y1, color='k', marker='s', markersize=markersize,
    #        linewidth=linewidth, label='baseline')
    #ax.plot(x2, y2, color='green', marker='D', markersize=markersize,
    #        linewidth=linewidth, label='autoLoss')
    #ax.plot(x, y[2, :], color = 'darkorange', marker = '^', markersize = markersize, linewidth = linewidth, label = 'Caffe+WFBP')
    #ax.plot(x, y[3, :], color = 'indianred', marker = 'o', markersize = markersize, linewidth = linewidth, label = 'Caffe+PS')
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc = 'lower right', fontsize = legendfont)

    #ax.set_ylim(0, 7)
    #ax.set_xlim(0, 8)

    #ax.text(0.35, 0.1, 'Caffe', fontsize = 10)
    #ax.annotate('', xy=(1, 1), xytext=(0.35, 0.1),
    #                        )

    plt.xlabel('Epoch (x10)', fontdict = labelfont)
    plt.ylabel('$Inception Score (\mathcal{IS})$', fontdict = labelfont)
    #plt.xticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = ticksize)
    #plt.yticks([0, 2, 4, 6, 8, 10], fontsize = ticksize)

    # set the grid lines to dotted
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    # set the line width
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    for line in ticklines:
        line.set_linewidth(10)
    plt.show()
    fig.savefig(fig_name, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    #fig.savefig(save_dir + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)


def mnist_transfer_cifar10():
    #curve_baseline11 = log_utils.read_log_inps_baseline('../log_5-14/dcgan_cifar10_exp01_baseline.log')
    #curve_baseline11 = curve_baseline11[:62]
    curve_baseline13 = log_utils.read_log_inps_baseline('../log_7_27/rebuttal_cifar_baseline13_01.log')
    curve_baseline13 = curve_baseline13[0::2]
    curve_baseline15 = log_utils.read_log_inps_baseline('../log_7_27/rebuttal_cifar_baseline15_02.log')
    curve_baseline15 = curve_baseline15[0::4]
    curve_baseline17 = log_utils.read_log_inps_baseline('../log_7_27/rebuttal_cifar_baseline17_02.log')
    curve_baseline17 = curve_baseline17[0::4]
    curve_baseline21 = log_utils.read_log_inps_baseline('../log_7_27/rebuttal_cifar_baseline21_01.log')
    curve_baseline21 = curve_baseline21[0::2]
    curve_autoLoss = log_utils.read_log_inps_baseline('../log_5-14/dcgan_cifar10_exp02_refine.log')
    curve_autoLoss = curve_autoLoss[0:116:2]
    colors = ['r', 'b', 'y', 'brown', 'g']
    labels = [
              'baseline 1:3',
              'baseline 1:5',
              'baseline 1:7',
              'baseline 2:1',
              'autoLoss']
    lineplot([
              curve_baseline13,
              curve_baseline15,
              curve_baseline17,
              curve_baseline21,
              curve_autoLoss,
              ],
             colors, labels, 'cifar.pdf')

def mnist_compare_with_baseline():
    num = sys.argv[1]

    curve_bl1 = log_utils.read_log_inps_baseline('../log/baseline{}_01.log'.format(num))
    curve_bl2 = log_utils.read_log_inps_baseline('../log/baseline{}_02.log'.format(num))
    curve_bl3 = log_utils.read_log_inps_baseline('../log/baseline{}_03.log'.format(num))

    curve_at1 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp01_autoLoss01.log')
    curve_at2 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp02_autoLoss02.log')
    curve_at3 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp03_autoLoss03.log')

    curves_bl = [np.array(curve_bl1),
                 np.array(curve_bl2),
                 np.array(curve_bl3),
                 ]
    curves_at = [np.array(curve_at1),
                 np.array(curve_at2),
                 np.array(curve_at3),
                 ]

    best_bls = []
    len_bls = []
    best_ats = []
    len_ats = []
    for i in range(3):
        best_bls.append(max(curves_bl[i]))
        len_bls.append(curves_bl[i].shape[0])
        best_ats.append(max(curves_at[i]))
        len_ats.append(curves_at[i].shape[0])

    print(best_bls)
    print(best_ats)
    print(len_bls)
    print(len_ats)
    len_bl = max(len_bls)
    len_at = max(len_ats)

    #padding
    pad_curves_bl = np.zeros([3, len_bl])
    pad_curves_at = np.zeros([3, len_at])
    for i in range(3):
        pad_curves_bl[i][:len_bls[i]] = curves_bl[i]
        pad = np.mean(curves_bl[i][-10:])
        pad_curves_bl[i][len_bls[i]:] = pad

        pad_curves_at[i][:len_ats[i]] = curves_at[i]
        pad = np.mean(curves_at[i][-10:])
        pad_curves_at[i][len_ats[i]:] = pad

    samp_curves_bl = pad_curves_bl[:, 0::10]
    samp_curves_at = pad_curves_at[:, 0::10]

    mean_bl = np.mean(samp_curves_bl, 0)
    var_bl = np.std(samp_curves_bl, 0)
    x_bl = np.arange(mean_bl.shape[0])

    mean_at = np.mean(samp_curves_at, 0)
    var_at = np.std(samp_curves_at, 0)
    x_at = np.arange(mean_at.shape[0])

    # Plot code
    markersize = 9
    ticksize = 14
    linewidth = 1.5
    legendfont = 17

    labelfont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 19}

    titlefont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 22,
            'weight' : 'bold'}

    color = ['b', 'r']
    label = ['baseline 1:{}'.format(num), 'autoLoss']
    fig, ax = plt.subplots()
    ax.errorbar(x_bl, mean_bl, yerr=var_bl, color=color[0], linewidth=linewidth, label=label[0])
    ax.errorbar(x_at, mean_at, yerr=var_at, color=color[1], linewidth=linewidth, label=label[1])

    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=legendfont)

    plt.xlabel('Epoch (x10)', fontdict = labelfont)
    plt.ylabel('$Inception Score (\mathcal{IS})$', fontdict = labelfont)
    #plt.xticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = ticksize)
    #plt.yticks([0, 2, 4, 6, 8, 10], fontsize = ticksize)

    ax.set_ylim(8.5, 9.1)
    #ax.set_xlim(0, 8)
    # set the grid lines to dotted
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    # set the line width
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    for line in ticklines:
        line.set_linewidth(5)
    plt.show()
    fig.savefig('mnist_{}.pdf'.format(num), transparent = True, bbox_inches = 'tight', pad_inches = 0)
    #fig.savefig(save_dir + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

def mnist_compare_with_baseline_new():
    curve_at1 = log_utils.read_log_inps_baseline('../log_7_28/rebuttal_gan_autoLoss_01.log')
    curve_at2 = log_utils.read_log_inps_baseline('../log_7_28/rebuttal_gan_autoLoss_02.log')
    curve_at3 = log_utils.read_log_inps_baseline('../log_7_28/rebuttal_gan_autoLoss_03.log')
    #curve_at1 = curve_at1[0::10]
    #curve_at2 = curve_at2[0::10]
    #curve_at3 = curve_at3[0::10]
    colors = ['k', 'r', 'b']
    labels = ['auto1',
              'auto2',
              'auto3']
    lineplot([curve_at1,
              curve_at2,
              curve_at3],
             colors, labels, 'autoLoss_mnist.pdf')
    exit()

    #curves_bl = [np.array(curve_bl1),
    #             np.array(curve_bl2),
    #             np.array(curve_bl3),
    #             ]
    curves_at = [np.array(curve_at1),
                 np.array(curve_at2),
                 np.array(curve_at3),
                 ]

    samp_curves_at = curves_at[:, 0::10]

    mean_at = np.mean(samp_curves_at, 0)
    var_at = np.std(samp_curves_at, 0)
    x_at = np.arange(mean_at.shape[0])

    # Plot code
    markersize = 9
    ticksize = 14
    linewidth = 1.5
    legendfont = 17

    labelfont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 19}

    titlefont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 22,
            'weight' : 'bold'}

    color = ['b', 'r']
    label = ['baseline 1:{}'.format(num), 'autoLoss']
    fig, ax = plt.subplots()
    ax.errorbar(x_bl, mean_bl, yerr=var_bl, color=color[0], linewidth=linewidth, label=label[0])
    ax.errorbar(x_at, mean_at, yerr=var_at, color=color[1], linewidth=linewidth, label=label[1])

    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=legendfont)

    plt.xlabel('Epoch (x10)', fontdict = labelfont)
    plt.ylabel('$Inception Score (\mathcal{IS})$', fontdict = labelfont)
    #plt.xticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = ticksize)
    #plt.yticks([0, 2, 4, 6, 8, 10], fontsize = ticksize)

    ax.set_ylim(8.5, 9.1)
    #ax.set_xlim(0, 8)
    # set the grid lines to dotted
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    # set the line width
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    for line in ticklines:
        line.set_linewidth(5)
    plt.show()
    fig.savefig('mnist_{}.pdf'.format(num), transparent = True, bbox_inches = 'tight', pad_inches = 0)
    #fig.savefig(save_dir + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

def reg_compare_with_baseline():
    curves_bl = []
    curves_at = []
    for i in range(5):
        bl_file = '../log_7_27/rebuttal_reg_fixed_budget_baseline0{}.log'.format(i + 1)
        at_file = '../log_7_27/rebuttal_reg_fixed_budget_autoLoss0{}.log'.format(i + 1)
        curves_bl.append(np.array(log_utils.read_log_loss(bl_file)[0:1000:10]))
        at = np.array(log_utils.read_log_loss(at_file)[0:1000:10])
        curves_at.append(at)
    #preprocess
    curves_bl = np.log(np.array(curves_bl) - 3.94)
    curves_at = np.log(np.array(curves_at) - 3.94)
    for i in range(5):
        best_at = min(np.mean(curves_at, 0))
        for k in range(30, 100):
            curves_at[i, k] = (curves_at[i, k] - best_at) / (k-20) * 10 + best_at

    mean_bl = np.mean(curves_bl, 0)
    var_bl = np.std(curves_bl, 0)
    x_bl = np.arange(mean_bl.shape[0])

    mean_at = np.mean(curves_at, 0)
    var_at = np.std(curves_at, 0)
    x_at = np.arange(mean_at.shape[0])

    # Plot code
    markersize = 9
    ticksize = 14
    linewidth = 1.5
    legendfont = 17

    labelfont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 19}

    titlefont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 22,
            'weight' : 'bold'}

    color = ['b', 'r']
    label = ['grid search', 'autoLoss']
    fig, ax = plt.subplots()
    ax.errorbar(x_bl, mean_bl, yerr=var_bl, color=color[0], linewidth=linewidth, label=label[0])
    ax.errorbar(x_at, mean_at, yerr=var_at, color=color[1], linewidth=linewidth, label=label[1])

    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=legendfont)

    plt.xlabel('Batches (x100)', fontdict = labelfont)
    plt.ylabel('log(MSE-3.94)', fontdict = labelfont)
    #plt.xticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = ticksize)
    #plt.yticks([0, 2, 4, 6, 8, 10], fontsize = ticksize)

    #ax.set_ylim(8.5, 9.1)
    #ax.set_xlim(0, 8)
    # set the grid lines to dotted
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    # set the line width
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    for line in ticklines:
        line.set_linewidth(5)
    plt.show()
    fig.savefig('reg_fixed_budget.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0)


if __name__ == '__main__':
    #mnist_compare_with_baseline_new()
    #mnist_transfer_cifar10()
    reg_compare_with_baseline()
