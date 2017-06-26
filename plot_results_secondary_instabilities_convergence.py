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
experiments = [paraL96_2lay, paraL96_1lay]

testzeroclv=True
steplengthforsecondorder = np.arange(0,15,3)
hs=[1.0, 0.5] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]



plt.figure()
for paraL96,h in product(experiments,hs):
    if not paraL96['2lay'] and h == 0.5: break
    savename=paraL96['expname']+"_h_"+str(h)
    spinup = paraL96['spinup']        
    paraL96=np.load(savename+"/paraL96.npy")
    paraL96=paraL96[()]
    
    tendcorr = np.memmap(savename+'/tendcorr.dat',mode='r',shape=(len(paraL96['time'])),dtype='float64')
    if paraL96['2lay']:
        legendentry = '2 lay, h='+str(paraL96['h'])+', b='+str(paraL96['b'])+', c='+str(paraL96['c'])
        line = '-'
    else:
        legendentry = '1 lay, h='+str(paraL96['h'])
        line = ':'

    color = 'k' if h == 1.0 else 'r'
    
    plt.plot(paraL96['time'][0::10],np.abs(savitzky_golay(tendcorr[0::10],1,0)), color+line,label = legendentry)
plt.legend(fontsize='small', loc = 'lower center')
plt.tight_layout()
plt.xlabel('Time [MTU]')
plt.ylabel('Correlation')  
plt.xticks(range(0,1001,100))
plt.ylim(0,1.0)

ax= plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)         
plt.title('Correlation of the CLV\n with the nearest to zero exponent with the tendency')
plt.tight_layout()
plt.savefig(resultsfolder+"/corrtend.pdf")
plt.savefig(resultsfolder+"/corrtend.png")