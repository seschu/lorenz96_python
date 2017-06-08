#!/home/user/anaconda3/bin/python
import numpy as np
from l96 import *
import os

# these are our constants
paraL96 = {'F1' : 64,
           'F2' : 0,
           'b'  : 10,
           'c'  : 10,
           'h'  : 1,
           'dimX': 20,
           'dimY' : 10,
           'RescaledY' : False
           }

# M number exponents
M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
integrator = 'classic'
spinup = 50;
#setup L96
dts = [0.01]
hs=[1] #[ 0.    ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
lyapmean_blv = np.zeros((M,len(dts)))

condnb = []
tmax=50

for countdt,dt_min in enumerate(dts):
    condnb.append(np.zeros(1))
    lyaploc_blv = np.zeros((1,M))
    for counti,rescale in enumerate([False]):
        paraL96['RescaledY']=rescale
        # Compute the exponents
    
        for count,h in enumerate(hs):
            t = np.zeros(1)
            paraL96['h']=h
            print("\nExperiment is the following:")
            for key in paraL96.keys(): print(key+' : '+str(paraL96[key]))
            L96,L96Jac,L96JacV,L96JacFull,dimN = setupL96_2layer(paraL96)
            field = GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=integrator)
            # initialize fields 
            print("\nInitialize ...")
            field.init_back('random',0.1)
            field.init_lin('random',0.1)
            field.restoreQ()   
            # spinup
            print("\nSpinup ...")
            #for i in range(0,int(spinup/0.1),1): 
            field.integrate_back(spinup)
            lyap = np.zeros(M)
            field.step_t = 0.0
            print("\nQR Steps ...")
            # Do QR step
            tn2=-1
            tn=0
            while field.step_t < tmax: 
                field.qr_step(dt_min,condnb_max=10**4)
                print(tn,field.step,field.step_t,field.condnb)
                t=np.insert(t,tn,field.step_t,axis=0)
                lyaploc_blv=np.insert(lyaploc_blv,tn,field.lyap,axis=0)
                tn=tn+1
            lyapmean_blv[:,countdt]=np.mean(lyaploc_blv[int((tn)*0.1):,:],axis=0)


savename='condnbmax'
print("Saveing results in folder "+savename+".")

if not os.path.exists(savename): os.mkdir(savename)
np.save(savename+'/condnb',condnb)
np.save(savename+'/dts',dts)
np.save(savename+'/paraL96',paraL96)
