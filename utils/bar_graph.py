#!/usr/bin/env python
# a bar plot with errorbars
import matplotlib as mpl
import os
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 2.5))
# fig.subplots_adjust(left=0.2, bottom=0.2)
#ax = fig.add_subplot(111)


data = [
(0.6400, 0.8795, 0.9204),
(0.9675, 0.9813, 0.9863),
(0.9836, 0.9859, 0.9910)
]

N = 3

width = 0.06       # the width of the bars
margin = width * N + 0.08
start = 0.07
ind  = np.array([start, start + margin, start + 2*margin])
xstart = 0.13
xs = [xstart, xstart + margin, xstart + 2*margin ] 

labelfont = {
        'color':  'black',
        'weight': 'normal',
        'size': 16}

colors = ['m', 'darkorange', 'c']
rects = []
for i in range(N):
  m = ax.bar(ind + width * i, data[i], width, color=colors[i], edgecolor='black', hatch='x')
  rects.append(m)

# fig, ax = plt.subplots()
#rects11 = ax.bar(ind, compute_time, width, color='c', edgecolor='black', hatch='x')
#rects12 = ax.bar(ind, comm_time, width, color='red', edgecolor='black', hatch='', bottom=compute_time)
#rects21 = ax.bar(ind2, compute_time_singletable, width, color='c', edgecolor='black', hatch='x')
#rects22 = ax.bar(ind2, comm_time_singletable, width, color='red', edgecolor='black', hatch='', bottom=compute_time_singletable)
#rects31 = ax.bar(ind3, compute_time_baseline, width, color='c', edgecolor='black', hatch='x')
#rects32 = ax.bar(ind3, comm_time_baseline, width, color='red', edgecolor='black', hatch='', bottom=compute_time_baseline)

# add some
ax.set_xlim(xmin=0.0, xmax=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(('n=20', 'n=50', 'n=100'), fontdict = labelfont)
ax.set_ylabel('Accuracy', fontdict = labelfont)
ax.set_ylim(ymax=1.08, ymin=0.5)
ax.set_yticks([0.6, 0.8, 1.0])
ax.legend((rects[0][0], rects[1][0], rects[2][0]), ('CVAE-semi', 'TripleGAN', 'SGAN'), ncol=3, loc=1, prop={'size':8})


gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.tight_layout()
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#plt.show()
save_dir = os.path.join('/home/hao/Dropbox/Projects/NIPS17-semi-gan/figures/', 'controllability')
fig.savefig(save_dir + '.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0)
fig.savefig(save_dir + '.jpg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
