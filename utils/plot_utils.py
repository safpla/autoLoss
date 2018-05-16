import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import numpy.matlib
import matplotlib as mpl
import os

import log_utils
name = ''
engine = ''

def lineplot(data):
    x = np.array(range(data.shape[0]))
    y = data

    # Plot code
    markersize = 9
    ticksize = 14
    linewidth = 1.5
    legendfont = 17

    labelfont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 10}

    titlefont = {#'family': 'times',
            'color':  'black',
            'weight': 'normal',
            'size': 22,
            'weight' : 'bold'}

    fig, ax = plt.subplots()
    ax.plot(x, y, color='k', marker='s', markersize=markersize,
            linewidth=linewidth, label='baseline')
    #ax.plot(x, y[1, :], color = 'green', marker = 'D', markersize = markersize, linewidth = linewidth, label = 'Poseidon')
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
    plt.xticks(x, fontsize = ticksize)
    plt.yticks(x, fontsize = ticksize)
    plt.title( name + '(Transfer to Cifar10)', fontdict = titlefont)

    # set the grid lines to dotted
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    # set the line width
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    for line in ticklines:
        line.set_linewidth(10)
    #plt.show()
    fig.savefig('test.jpg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    #fig.savefig(save_dir + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)

if __name__ == '__main__':
    curve_baseline = log_utils.read_log_inps_baseline('../log_5-14/dcgan_cifar10_exp01_baseline.log')
    lineplot(np.array(curve_baseline))
