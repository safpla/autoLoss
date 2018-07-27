#!/usr/bin/env python
# a bar plot with errorbars
import matplotlib as mpl
import os
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

#fig, ax = plt.subplots(figsize=(5, 2.5))
fig, ax = plt.subplots()
# fig.subplots_adjust(left=0.2, bottom=0.2)
#ax = fig.add_subplot(111)


data = [
(0.6400, 0.8795, 0.9204),
(0.9675, 0.9813, 0.9863),
(0.9836, 0.9859, 0.9910)
]

#data = [(5.47, 2.48, 4.84, 5.46, 6.67, 8.53, 1.97, 6.61, 4.76, 7.16, 5.11, 5.77, 8.18, 7.62, 8.71, 8.01, 5.14, 8.09, 7.01, 2.69),
#        (5.93, 3.38, 5.50, 5.99, 5.90, 8.96, 3.29, 6.35, 5.29, 7.35, 5.70, 6.12, 8.43, 8.73, 8.77, 8.07, 5.75, 8.34, 7.61, 2.61)]
data = [(8.53, 7.16, 8.18, 7.62, 8.71, 8.01, 8.09, 7.02, 8.71, 8.72, 8.55, 7.12, 8.18, 8.74, 8.19, 7.04, 8.12, 7.14, 8.61, 8.38),
        (8.96, 7.35, 8.43, 8.73, 8.77, 8.07, 8.34, 7.61, 8.83, 8.39, 8.34, 7.14, 8.51, 8.25, 8.40, 7.53, 8.31, 7.89, 8.78, 7.41)]
xtl = []
for i in range(20):
    xtl.append(str(i))

N = 2

width = 0.1       # the width of the bars
margin = width * N + 0.08
start = 0.03
ind  = np.arange(20) * margin + start
xstart = 0.13
xs = ind.tolist()

labelfont = {
        'color':  'black',
        'weight': 'normal',
        'size': 21}

colors = ['darkorange', 'darkgreen']
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
#ax.set_xlim(xmin=0.0, xmax=0.8)
#ax.set_xticks(xs)
#ax.set_xticklabels(xtl, fontdict = labelfont)
ax.set_xlabel('DCGAN Architectures', fontdict=labelfont)
ax.set_ylabel('Inception Score ($\mathcal{IR}$)', fontdict = labelfont)
ax.set_ylim(ymax=9, ymin=6)
#ax.set_yticks([0.6, 0.8, 1.0])
ax.legend((rects[0][0], rects[1][0]), ('DCGAN', 'AutoLoss'), ncol=1, loc=2, prop={'size':12})

ax.grid(color = 'black', linestyle = '-.', linewidth = 1)
ax.grid(color = 'black', linestyle = '-.', linewidth = 1)

gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.xticks([], [])
plt.grid(True)
plt.tight_layout()
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()
save_dir = 'transfer_to_models'
fig.savefig(save_dir + '.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0)
