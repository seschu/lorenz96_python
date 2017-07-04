#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savitzky_golay
# these are our constants
paraL96_2lay = {'F1' : 10,
           'F2' : 0,
           'b'  : 10,
           'c'  : 10,
           'h'  : 1,
           'dimX': 36,
           'dimY' : 10,
           'RescaledY' : False,
           'expname' : 'secondaryinstabilities_2layer',
           'time' : np.arange(0,1000,0.1),
           'spinup' : 100,
           '2lay' : True
           }

paraL96_1lay = {'F1' : 10,
           'F2' : 0,
           'b'  : 10,
           'c'  : 10,
           'h'  : 1,
           'dimX': 44,
           'dimY' : 10,
           'RescaledY' : False,
           'expname' : 'secondaryinstabilities_1layer',
           'time' : np.arange(0,1000,0.1),
           'spinup' : 100,
           '2lay' : False
           }

resultsfolder = 'secondaryinstabilities'
if not os.path.exists(resultsfolder): os.mkdir(resultsfolder)
experiments = [paraL96_1lay]

testzeroclv=True
steplengthforsecondorder = np.arange(0,15,3)
hs=[1.0, 0.5] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]



f, axarr = plt.subplots(2, figsize = (7,6))

axarr[0].axhline(y=0,color='lightgrey')
axarr[1].axhline(y=0,color='lightgrey')
for paraL96,h in product(experiments,hs):
    if not paraL96['2lay'] and h == 0.5: break
    savename=paraL96['expname']+"_h_"+str(h)
    paraL96=np.load(savename+"/paraL96.npy")
    paraL96=paraL96[()]
    dimN = paraL96['dimX']  + paraL96['dimX']*paraL96['dimY'] if paraL96['2lay'] else paraL96['dimX']
    maskcorr=np.load(savename+"/maskcorr.npy")
    lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(paraL96['time']),dimN),dtype='float64')
    if paraL96['2lay']:
        legendentry = '2 lay, h='+str(paraL96['h'])+', b='+str(paraL96['b'])+', c='+str(paraL96['c'])
        line = ''
        subplot=0
    else:
        legendentry = '1 lay'
        line = ':'
        subplot=1

    color = 'k' if h == 1.0 else 'r'
    
    axarr[subplot].plot(range(1,dimN+1),np.memmap.mean(lyaploc_clv[maskcorr,:],axis = 0), color+line,label = legendentry,marker ='.',markersize=2,linestyle='none')
axarr[0].legend(fontsize='small', loc = 'upper right')
axarr[1].legend(fontsize='small', loc = 'upper right')
axarr[0].set_xlabel('Lyapunov Index')
axarr[1].set_xlabel('Lyapunov Index')
axarr[0].set_ylabel('Lyapunov Exponent [1/MTU]')  
axarr[1].set_ylabel('Lyapunov Exponent [1/MTU]')          
axarr[0].set_title('Lyapunov Spectra')
f.tight_layout()
f.savefig(resultsfolder+"/lyap.pdf")
f.savefig(resultsfolder+"/lyap.png")