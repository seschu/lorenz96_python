#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product 
import warnings
import matplotlib.pyplot as plt
#warnings.simplefilter("error")
#warnings.simplefilter("ignore", DeprecationWarning)

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


testzeroclv=True

hs=[ 0.25 ] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
experiments = [paraL96_2lay]

for paraL96,h in product(experiments,hs):
    if not paraL96['2lay'] and not h == 1.0: print("1 lay only with h = 1.");break
    # M number exponents
    if paraL96['2lay']:
        M = paraL96['dimX']  + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
        dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
    else:
        M = paraL96['dimX'] 
        dimN = paraL96['dimX'] 
        
    integrator = 'classic'
    t = paraL96['time']
    dt = np.mean(np.diff(t))
    
    savename=paraL96['expname']+"_h_"+str(h)
    spinup = paraL96['spinup']
    
    #setup L96
    
    
    
    if not os.path.exists(savename): os.mkdir(savename)
    CLV = np.memmap(savename+'\CLV.dat',mode='w+',shape=(len(t),dimN,M),dtype='float64')
    BLV = np.memmap(savename+'\BLV.dat',mode='w+',shape=(len(t),dimN,M),dtype='float64')
    R = np.memmap(savename+'\R.dat',mode='w+',shape=(len(t),dimN,M),dtype='float64')
    lyapmean_blv = np.memmap(savename+'\lyapmean_blv.dat',mode='w+',shape=(M),dtype='float64')
    lyapmean_clv = np.memmap(savename+'\lyapmean_clv.dat',mode='w+',shape=(M),dtype='float64')
    lyaploc_clv = np.memmap(savename+'\lyaploc_clv',mode='w+',shape=(len(t),M),dtype='float64')
    lyaploc_blv = np.memmap(savename+'\lyaploc_blv',mode='w+',shape=(len(t)-1,M),dtype='float64')
    np.save(savename+'\t',t)
    trajectory = np.memmap(savename+'\trajectory.dat',mode='w+',shape=(len(t),dimN),dtype='float64')
    if testzeroclv: tendency = np.memmap(savename+'\tendency.dat',mode='w+',shape=(len(t),dimN),dtype='float64')
    if testzeroclv: tendcorr = np.memmap(savename+'\tendcorr.dat',mode='w+',shape=(len(t)),dtype='float64')
    
    
    # Compute the exponents
    
    paraL96['h']=h
    print("\nExperiment is the following:")
    for key in paraL96.keys(): print(key+' : '+str(paraL96[key]))
    if paraL96['2lay']: L96,L96Jac,L96JacV,L96JacFull,dimN = l96.setupL96_2layer(paraL96)
    else: L96,L96Jac,L96JacV,L96JacFull,dimN = l96.setupL96(paraL96)
    field = l96.GinelliForward(dimN,M,tendfunc = L96, jacfunc = None, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=integrator)
    # initialize fields 
    print("\nInitialize ...")
    field.init_back('random',0.1)
    field.init_lin('random',0.1)
    field.restoreQ()   
    # spinup
    print("\nSpinup ...")

    field.integrate_back(spinup)    
    field.step_t = 0.0

    BLV[0,:,:]=field.x['lin']
    print("\nQR Steps ...")
    # Do QR step
    for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
        if testzeroclv: tendency[tn,:] = L96(ts,field.x['back'])
        trajectory[tn,:]=field.x['back']
        field.qr_step(te-ts,dt=dt)
        R[tn,:,:]=field.R
        BLV[tn+1,:,:]=field.x['lin']
        print(te)
        lyaploc_blv[tn,:]=field.lyap
        if tn % 100 == 0:
            np.memmap.flush(BLV)
            np.memmap.flush(R)
            np.memmap.flush(lyaploc_blv)
            if testzeroclv: np.memmap.flush(tendency)
    lyapmean_blv[:]=np.mean(lyaploc_blv[int(tn/2):,:],axis=0)
    
    # Do Backwards steps

    print("\nBackwards Steps ...")
    imax = tn
    res=triu(np.random.rand(M,M))
    CLV[imax,:,:]=np.matmul(BLV[imax,:,:],res)
    res=res/np.linalg.norm(res,axis=0,keepdims=True)
    lyaploc_clv[imax,:]=np.log(1/np.abs(np.linalg.norm(CLV[imax,:,:],axis=0)))/np.abs(te-ts)

    if testzeroclv: minloc=np.argmin(np.abs(lyapmean_blv))
        
    for tn, (ts,te) in enumerate(zip(t[-2:0:-1],t[-1:0:-1])):
        n = imax-tn
        if n==0: break
        res=linalg.solve(R[n-1,0:M,:],res)
        CLV[n-1,:,:]=np.matmul(BLV[n-1,:,:],res)
        res=res/np.linalg.norm(res,axis=0,keepdims=True)
        lyaploc_clv[n-1,:]=np.log(1/np.abs(np.linalg.norm(CLV[n-1,:,:],axis=0)))/(te-ts)
        CLV[n-1,:,:]= CLV[n-1,:,:]/np.linalg.norm(CLV[n-1,:,:],axis=0,keepdims=True)
        
        if testzeroclv:
            tendcorr[n-1]=np.sum(np.multiply(CLV[n-1,:,minloc],tendency[n-1,:]))/np.sqrt(np.sum(np.multiply(tendency[n-1,:],tendency[n-1,:])))
            
        if tn % 100 == 0:
            np.memmap.flush(R)
            np.memmap.flush(CLV)
            np.memmap.flush(lyaploc_clv)
            np.memmap.flush(tendcorr)
    lyapmean_clv[:]=np.mean(lyaploc_clv[int(tn/2):,:],axis=0)
    
    
    
    print("Saveing results in folder "+savename+".")
    np.save(savename+"/paraL96",paraL96)
    np.save(savename+"/h",h)
    
    invCLV = np.memmap(savename+'\invCLV.dat',mode='w+',shape=(len(t),dimN,M),dtype='float64')
    
    for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
        invCLV[tn,:,:]=np.linalg.inv(CLV[tn,:,:])  
    
    corrs=np.arange(0.01,1,0.1)
    lowest=np.zeros((corrs.shape[0],M))
    length=np.zeros((corrs.shape[0],M))
    for n,c in enumerate(corrs):
        d = (np.logical_and(np.abs(tendcorr[:])>c,np.abs(tendcorr[:])<c+0.1))
        if d.any(): 
            length[n,:] = np.sum(d) 
            lowest[n,:] = np.average(lyaploc_clv[:,:],weights = d, axis = 0)
        else:
            lowest[n,:] = 0
    maskcorr = (np.logical_and(np.abs(tendcorr[:])>0.95,np.abs(tendcorr[:])<1.01))
    np.save(savename+"/maskcorr",maskcorr )

