#!/home/user/anaconda3/bin/python
import numpy as np
from l96 import *

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
dts = [0.005]
hs=[1.0] #[ 0.    ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
lyapmean_blv = np.zeros((M,len(dts)))

condnb = []


for countdt,dt in enumerate(dts):
    t = np.arange(0,20*dt/0.01,dt)
    condnb.append(np.zeros(len(t)+1))
    lyaploc_blv = np.zeros((len(t)-1,M))
    for counti,rescale in enumerate([False]):
        paraL96['RescaledY']=rescale
        # Compute the exponents
    
        for count,h in enumerate(hs):
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
            
            for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
                condnb[countdt][tn+1]=field.qr_step(te-ts)
                print(te,dt)
                lyaploc_blv[tn,:]=field.lyap
            lyapmean_blv[:,countdt]=np.mean(lyaploc_blv[int(tn*0.1):,:],axis=0)


savename='condnb'
print("Saveing results in folder "+savename+".")

if not os.path.exists(savename): os.mkdir(savename)
np.save(savename+'/condnb',condnb)
np.save(savename+'/dts',dts)
np.save(savename+'/paraL96',paraL96)
np.save(savename+'/lyapmean_blv',lyapmean_blv)
