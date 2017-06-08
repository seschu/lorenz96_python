#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'medium',
          'figure.figsize': (6, 3),
         'axes.labelsize': 'medium',
         'axes.titlesize':'large',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small'}
pylab.rcParams.update(params)

savename='definiterun'

print("Loading results from folder "+savename+".")

lyaploc_blv=np.load(savename+'/lyaploc_blv.npy')
lyaploc_clv=np.load(savename+'/lyaploc_clv.npy')
lyapmean_blv=np.load(savename+'/lyapmean_blv.npy')
lyapmean_clv=np.load(savename+'/lyapmean_clv.npy')
paraL96=np.load(savename+'/paraL96.npy')
paraL96=paraL96[()]
M=np.load(savename+'/M.npy')
imax=np.load(savename+'/imax.npy')
hs=np.load(savename+'/h.npy')
CLV=np.load(savename+'/CLV.npy')

start=CLV.shape[0]*0.2
ende=CLV.shape[0]*0.8
figurename='lyapspec.png'#'lyapspec_rescaled_vs_scale.pdf'

print("Printing to figure "+figurename+".")


posLE=lambda X:X[np.where(X>0)].shape[0]
def dKY(X):
    res=X[0]
    i=0
    while res >=0:
        i=i+1
        res=res + X[i]
    res = i + (res-X[i])/np.abs(X[i])
    return res

fig, ax = plt.subplots(1, 3,  sharex='col',figsize=(20,8))    
for count,h in enumerate(hs):
    Y = lyapmean_blv[:,count]
    ax[0].plot(np.arange(1,M+1,1), Y, label = 'h = '+str(h)+'; posLE = '+str(posLE(Y))+'; dKY = '+"{0:.2f}".format(dKY(Y)).format())
    
ax[0].set_xlabel('Lyapunov Index')
ax[0].set_ylabel('Lyapunov Exponents [1/MTU]')
ax[0].set_title('BLE; Rescaled Y modes = '+str(paraL96['RescaledY'])+' and b = '+str(paraL96['b'])+', c = '+str(paraL96['c']))
ax[0].set_ylim([-75,50]) 
legend = ax[0].legend(loc='best', shadow=False, labelspacing = 0.5,frameon=False,borderaxespad=0.1,handleheight=0.0,borderpad=0.5)
for label in legend.get_texts():
    label.set_fontsize('medium')
frame = legend.get_frame()
frame.set_facecolor('1.0')

for count,h in enumerate(hs):
    Y = np.mean(lyaploc_clv[int(start):int(ende),:,count],axis = 0)
    ax[1].plot(np.arange(1,M+1,1),Y, label = 'h = '+str(h)+'; posLE = '+str(posLE(Y))+'; dKY = '+"{0:.2f}".format(dKY(Y)).format())
    
ax[1].set_xlabel('Lyapunov Index')
ax[1].set_ylabel('Lyapunov Exponents [1/MTU]')
ax[1].set_title('CLE; Rescaled Y modes = '+str(paraL96['RescaledY'])+' and b = '+str(paraL96['b'])+', c = '+str(paraL96['c']))
ax[1].set_ylim([-75,50]) 
legend = ax[1].legend(loc='best', shadow=False, labelspacing = 0.5,frameon=False,borderaxespad=0.1,handleheight=0.0,borderpad=0.5)
for label in legend.get_texts():
    label.set_fontsize('medium')
frame = legend.get_frame()
frame.set_facecolor('1.0')

for count,h in enumerate(hs):
    R = np.sum(np.mean(CLV[int(start):int(ende),0:paraL96['dimX'],:,count]**2,axis=0),axis=0)
    p=ax[2].plot(np.arange(1,M+1),R, label = 'h = '+str(h))
    ax[2].axvline(np.argmin(np.abs(np.mean(lyaploc_clv[int(start):int(ende),:,count],axis = 0)))+1,linewidth=1, color=p[0].get_color(), ls = ':')
    
ax[2].set_xlabel('Lyapunov Index')
ax[2].set_ylabel('Projection on X-Modes')
ax[2].set_title('CLV X-Modes Norm; Rescaled Y modes = '+str(paraL96['RescaledY'])+' and b = '+str(paraL96['b'])+', c = '+str(paraL96['c']))
ax[2].set_ylim([0,1.005])
legend = ax[2].legend(loc='best', shadow=False, labelspacing = 0.5,frameon=False,borderaxespad=0.1,handleheight=0.0,borderpad=0.5)
for label in legend.get_texts():
    label.set_fontsize('medium')
frame = legend.get_frame()
frame.set_facecolor('1.0')

fig.tight_layout()
fig.savefig(savename+"/"+figurename)

