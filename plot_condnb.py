#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'small',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'x-small',
         'xtick.labelsize':'x-small',
         'ytick.labelsize':'x-small'}
pylab.rcParams.update(params)

savename='condnb'

print("Loading results from folder "+savename+".")

condnb=np.load(savename+'/condnb.npy')
dts=np.load(savename+'/dts.npy')
paraL96=np.load(savename+'/paraL96.npy')

figurename='condnb.png'#'lyapspec_rescaled_vs_scale.pdf'

print("Printing to figure "+figurename+".")


fig,ax = plt.subplots(len(condnb),figsize=(5,10))    
for count,dt in enumerate(dts):
    n, bins, patches = ax[count].hist(condnb[count][1:-2], normed=1, facecolor='green', alpha=0.75, label = 'dt: '+str(dt),bins=np.logspace(np.log10(np.min(condnb[count][1:-2])), np.log10(np.max(condnb[count][1:-2])), 50))
    ax[count].set_xscale("log")
    ax[count].set_xlabel('Conditionnumber (\sigma_highest/\sigma_lowest)')
    ax[count].set_ylabel('relative probability')
    ax[count].set_title('Distribution of Condition number with time step = '+str(dt))
    #legend = ax[count].legend(loc='best', shadow=False,fontsize=6, labelspacing = 0.5,frameon=False,borderaxespad=0.1,handleheight=0.0,borderpad=0.5)
    #for label in legend.get_texts():
    #    label.set_fontsize('x-small')
    #frame = legend.get_frame()
    #frame.set_facecolor('1.0')
fig.tight_layout()
fig.savefig(savename+"/"+figurename)

