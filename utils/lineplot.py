import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import matplotlib as mpl
import os

name = 'VGG19-22K'
engine = 'caffe'


data = np.array([[18.5, 18.7, 18.9, 19.3, 19.4, 19.7, 20.1],
[18.5, 19.4, 20.8, 22.58, 24.2, 25.6, 27.7],
[18.5, 34.5, 36.8, 50.3, 68.5, 105, 134]])/20;
nodes = [1, 2, 4, 8, 16, 32]
# convert to throughput
indices = np.concatenate([[0], np.arange(2,data.shape[1])], axis = 0)
tmp = data[:, indices]
# convert per-iteration time to speedups
throughput = (1 / tmp) * np.matlib.repmat(nodes, tmp.shape[0], 1)
speedups = throughput / np.matlib.repmat(throughput[:, 0], tmp.shape[1], 1).T
ideal = nodes 
speedups = np.vstack((ideal, speedups))
print(speedups)
x = nodes
y = speedups


# Plot code

markersize = 9
ticksize = 14
linewidth = 1.5
legendfont = 17

labelfont = {#'family': 'times',
        'color':  'black',
        'weight': 'normal',
        'size': 20}

titlefont = {#'family': 'times',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        'weight' : 'bold'}

fig, ax = plt.subplots()
ax.plot(x, y[0, :], color = 'k', marker = 's', markersize = markersize, linewidth = linewidth, label = 'Linear')
ax.plot(x, y[1, :], color = 'green', marker = 'D', markersize = markersize, linewidth = linewidth, label = 'Poseidon')
ax.plot(x, y[2, :], color = 'darkorange', marker = '^', markersize = markersize, linewidth = linewidth, label = 'Caffe+WFBP')
ax.plot(x, y[3, :], color = 'indianred', marker = 'o', markersize = markersize, linewidth = linewidth, label = 'Caffe+PS')
ax.grid(True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = 'upper left', fontsize = legendfont)

ax.set_ylim(0, np.max(nodes) + 0.5)
ax.set_xlim(0, np.max(nodes) + 0.5)

#ax.text(0.35, 0.1, 'Caffe', fontsize = 10)
#ax.annotate('', xy=(1, 1), xytext=(0.35, 0.1),
#                        )


plt.xlabel('# of Nodes', fontdict = labelfont)
plt.ylabel('Speedups', fontdict = labelfont)
plt.xticks(x, fontsize = ticksize)
plt.yticks(x, fontsize = ticksize)
plt.title( name + ' (40 GbE)', fontdict = titlefont)

# set the grid lines to dotted
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

# set the line width
ticklines = ax.get_xticklines() + ax.get_yticklines()
for line in ticklines:
    line.set_linewidth(10)
#plt.show()
save_dir = os.path.join('/home/hao/Dropbox/projects/Poseidon/v2.0/ATC2017/pictures', 'scalability_' + engine + '_' + name)
fig.savefig(save_dir + '.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0)
#fig.savefig(save_dir + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
