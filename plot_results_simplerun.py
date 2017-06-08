#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'small',
          'figure.figsize': (5, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'x-small',
         'xtick.labelsize':'x-small',
         'ytick.labelsize':'x-small'}
pylab.rcParams.update(params)

savename='simple_run'

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
h=hs[0]
CLV=np.load(savename+'/CLV.npy')
#BLV=np.load(savename+'/BLV.npy')
#R=np.load(savename+'/R.npy')


start=0.2
ende=0.9

figurename='lyapspec.png'#'lyapspec_rescaled_vs_scale.pdf'


print("Printing to figure "+figurename+".")


posLE=lambda X:X[np.where(X>0)].shape[0]
def dKY(X):
    res=X[0]
    i=0
    while res >=0 and i+1<X.shape[0]:
        i=i+1
        res=res + X[i]
    res = i + (res-X[i])/np.abs(X[i])
    return res

fig, ax = plt.subplots(3, 1,  sharex='col',figsize=(10,15))    
Y = lyapmean_blv[:]
ax[0].plot(np.arange(1,M+1,1), Y, label = 'h = '+str(h)+'; posLE = '+str(posLE(Y))+'; dKY = '+"{0:.2f}".format(dKY(Y)).format())
    
ax[0].set_xlabel('Lyapunov Index')
ax[0].set_ylabel('Lyapunov Exponents [1/MTU]')
ax[0].set_title('BLE; Rescaled Y modes = '+str(paraL96['RescaledY'])+' and b = '+str(paraL96['b'])+', c = '+str(paraL96['c']))
legend = ax[0].legend(loc='best', shadow=False,fontsize=6, labelspacing = 0.5,frameon=False,borderaxespad=0.1,handleheight=0.0,borderpad=0.5)
for label in legend.get_texts():
    label.set_fontsize('x-small')
frame = legend.get_frame()
frame.set_facecolor('1.0')

Y = np.mean(lyaploc_clv[int(imax*start):int(imax*ende),:],axis = 0)
ax[1].plot(np.arange(1,M+1,1),Y, label = 'h = '+str(h)+'; posLE = '+str(posLE(Y))+'; dKY = '+"{0:.2f}".format(dKY(Y)).format())
    
ax[1].set_xlabel('Lyapunov Index')
ax[1].set_ylabel('Lyapunov Exponents [1/MTU]')
ax[1].set_title('CLE; Rescaled Y modes = '+str(paraL96['RescaledY'])+' and b = '+str(paraL96['b'])+', c = '+str(paraL96['c']))
legend = ax[1].legend(loc='best', shadow=False,fontsize=6, labelspacing = 0.5,frameon=False,borderaxespad=0.1,handleheight=0.0,borderpad=0.5)
for label in legend.get_texts():
    label.set_fontsize('x-small')
frame = legend.get_frame()
frame.set_facecolor('1.0')

ax[2].plot(np.arange(1,M+1),np.sum(np.mean(CLV[int(imax*start):int(imax*ende),0:paraL96['dimX'],:]**2,axis=0),axis=0), 
      label = 'h = '+str(h))
    
ax[2].set_xlabel('Lyapunov Index')
ax[2].set_ylabel('Projection on X-Modes')
ax[2].set_title('CLV X-Modes Norm; Rescaled Y modes = '+str(paraL96['RescaledY'])+' and b = '+str(paraL96['b'])+', c = '+str(paraL96['c']))
legend = ax[2].legend(loc='best', shadow=False,fontsize=6, labelspacing = 0.5,frameon=False,borderaxespad=0.1,handleheight=0.0,borderpad=0.5)
for label in legend.get_texts():
    label.set_fontsize('x-small')
frame = legend.get_frame()
frame.set_facecolor('1.0')
    
fig.tight_layout()
fig.savefig(savename+"/"+figurename)

