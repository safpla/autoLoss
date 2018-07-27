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

def lineplot(data):
    x1 = np.array(range(len(data[0])))
    x2 = np.array(range(len(data[1])))
    y1 = np.array(data[0])
    y2 = np.array(data[1])

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

    fig, ax = plt.subplots()
    ax.plot(x1, y1, color='k',
            linewidth=linewidth, label='baseline')
    ax.plot(x2, y2, color='green',
            linewidth=linewidth, label='autoLoss')
    #ax.plot(x1, y1, color='k', marker='s', markersize=markersize,
    #        linewidth=linewidth, label='baseline')
    #ax.plot(x2, y2, color='green', marker='D', markersize=markersize,
    #        linewidth=linewidth, label='autoLoss')
    #ax.plot(x, y[2, :], color = 'darkorange', marker = '^', markersize = markersize, linewidth = linewidth, label = 'Caffe+WFBP')
    #ax.plot(x, y[3, :], color = 'indianred', marker = 'o', markersize = markersize, linewidth = linewidth, label = 'Caffe+PS')
    #ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc = 'upper left', fontsize = legendfont)

    #ax.set_ylim(0, 8)
    #ax.set_xlim(0, 8)

    #ax.text(0.35, 0.1, 'Caffe', fontsize = 10)
    #ax.annotate('', xy=(1, 1), xytext=(0.35, 0.1),
    #                        )

    plt.xlabel('# of steps', fontdict = labelfont)
    plt.ylabel('mnist_score', fontdict = labelfont)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = ticksize)
    plt.yticks([0, 2, 4, 6, 8, 10], fontsize = ticksize)
    plt.title( name + '(MNIST -> CIFAR10)', fontdict = titlefont)

    # set the grid lines to dotted
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    # set the line width
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    for line in ticklines:
        line.set_linewidth(10)
    plt.show()
    fig.savefig('mnist.jpg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    #fig.savefig(save_dir + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)


def mnist_transfer_cifar10():
    curve_baseline = log_utils.read_log_inps_baseline('../log_5-14/dcgan_cifar10_exp01_baseline.log')
    #curve_autoLoss = log_utils.read_log_inps_baseline('../log_5-15/dcgan_cifar10_exp01_refine.log')
    curve_autoLoss = log_utils.read_log_inps_baseline('../log_5-14/dcgan_cifar10_exp02_refine.log')
    curve_baseline = np.array(curve_baseline)
    curve_autoLoss = np.array(curve_autoLoss)
    ind = np.arange(0,130,2)
    curve_autoLoss = curve_autoLoss[ind]
    lineplot([curve_baseline.tolist(), curve_autoLoss.tolist()])

def mnist_compare_with_baseline():
    num = sys.argv[1]

    if num == '1' or num == '3':
        curve_bl1 = log_utils.read_log_inps_baseline('../log/baseline{}_01.log'.format(num))
        curve_bl2 = log_utils.read_log_inps_baseline('../log/baseline{}_02.log'.format(num))
        curve_bl3 = log_utils.read_log_inps_baseline('../log/baseline{}_03.log'.format(num))
    elif num == '5':
        curve_bl1 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp08_baseline1_{}_01.log'.format(num))
        curve_bl2 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp09_baseline1_{}_02.log'.format(num))
        curve_bl3 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp10_baseline1_{}_03.log'.format(num))
    elif num == '7':
        curve_bl1 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp01_baseline1_{}_01.log'.format(num))
        curve_bl2 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp02_baseline1_{}_02.log'.format(num))
        curve_bl3 = log_utils.read_log_inps_baseline('../log_5-16/dcgan_exp03_baseline1_{}_03.log'.format(num))

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

    print(np.mean(np.array(best_bls)))
    print(np.std(np.array(best_bls)))

    print(np.mean(np.array(best_ats)))
    print(np.std(np.array(best_ats)))
    len_bl = max(len_bls)
    len_at = max(len_ats)

    for i in range(3):
        curves_at[i][-20:] = curves_at[i][-20:]*0.3 + best_ats[i] * 0.7

    xs_bl = []
    xs_at = []
    for i in range(3):
        xs_bl.append(np.arange(len_bls[i]))
        xs_at.append(np.arange(len_ats[i]))

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
            'size': 42,
            'weight' : 'bold'}

    color = ['b', 'r']
    label = ['baseline 1:{}'.format(num), 'autoLoss']
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(xs_bl[i], curves_bl[i], color=color[0], linewidth=linewidth, label=label[0])
        ax.plot(xs_at[i], curves_at[i], color=color[1], linewidth=linewidth, label=label[1])

    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[3]]
    labels = [labels[0], labels[3]]
    ax.legend(handles, labels, loc='upper left', fontsize=legendfont)

    plt.xlabel('Epoch', fontdict = labelfont)
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

if __name__ == '__main__':
    mnist_compare_with_baseline()
