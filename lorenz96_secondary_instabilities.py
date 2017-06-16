#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg

# these are our constants
paraL96 = {'F1' : 10,
           'F2' : 0,
           'b'  : 10,
           'c'  : 10,
           'h'  : 1,
           'dimX': 36,
           'dimY' : 10,
           'RescaledY' : False
           }

# M number exponents
M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
integrator = 'classic'
dt = 0.01 #np.mean(np.diff(t))
t = np.arange(0,1000,1)

spinup = 100;
#setup L96
hs=[ 1. ] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]


testzeroclv=True

savename='secondaryinstabilities_1layer'
if not os.path.exists(savename): os.mkdir(savename)
CLV = np.memmap(savename+'/CLV.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV = np.memmap(savename+'/BLV.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R = np.memmap(savename+'/R.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='w+',shape=(M,len(hs)),dtype='float64')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='w+',shape=(M,len(hs)),dtype='float64')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='w+',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='w+',shape=(len(t)-1,M,len(hs)),dtype='float64')
np.save(savename+'/t',t)
if testzeroclv: tendency = np.memmap(savename+'/tendency.dat',mode='w+',shape=(len(t),dimN,len(hs)),dtype='float64')
if testzeroclv: tendcorr = np.memmap(savename+'/tendcorr.dat',mode='w+',shape=(len(t),len(hs)),dtype='float64')


# Compute the exponents

for count,h in enumerate(hs):
    paraL96['h']=h
    print("\nExperiment is the following:")
    for key in paraL96.keys(): print(key+' : '+str(paraL96[key]))
    L96,L96Jac,L96JacV,L96JacFull,dimN = l96.setupL96_2layer(paraL96)
    field = l96.GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=integrator)
    # initialize fields 
    print("\nInitialize ...")
    field.init_back('random',0.1)
    field.init_lin('random',0.1)
    field.restoreQ()   
    # spinup
    print("\nSpinup ...")
    #for i in range(0,int(spinup/0.1),1): 
    field.integrate_back(spinup,dt=dt)
    field.step_t = 0.0

    BLV[0,:,:,count]=field.x['lin']
    print("\nQR Steps ...")
    # Do QR step
    for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
        if testzeroclv: tendency[tn,:,count] = L96(ts,field.x['back'])
        field.qr_step(te-ts,dt=dt)
        R[tn,:,:,count]=field.R
        BLV[tn+1,:,:,count]=field.x['lin']
        print(te,'|',count+1,'/',len(hs))
        lyaploc_blv[tn,:,count]=field.lyap
        if tn % 1 == 0:
            np.memmap.flush(BLV)
            np.memmap.flush(R)
            np.memmap.flush(lyaploc_blv)
            if testzeroclv: np.memmap.flush(tendency)
    lyapmean_blv[:,count]=np.mean(lyaploc_blv[int(tn/2):,:,count],axis=0)
    
    # Do Backwards steps

    print("\nBackwards Steps ...")
    imax = tn
    res=triu(np.random.rand(M,M))
    CLV[imax,:,:,count]=np.matmul(BLV[imax,:,:,count],res)
    res=res/np.linalg.norm(res,axis=0,keepdims=True)
    lyaploc_clv[imax,:,count]=np.log(1/np.abs(np.linalg.norm(CLV[imax,:,:,count],axis=0)))/np.abs(te-ts)

    if testzeroclv: minloc=np.argmin(np.abs(lyapmean_blv))
        
    for tn, (ts,te) in enumerate(zip(t[-2:0:-1],t[-1:0:-1])):
        n = imax-tn
        if n==0: break
        res=linalg.solve(R[n-1,0:M,:,count],res)
        CLV[n-1,:,:,count]=np.matmul(BLV[n-1,:,:,count],res)
        res=res/np.linalg.norm(res,axis=0,keepdims=True)
        lyaploc_clv[n-1,:,count]=np.log(1/np.abs(np.linalg.norm(CLV[n-1,:,:,count],axis=0)))/(te-ts)
        CLV[n-1,:,:,count]= CLV[n-1,:,:,count]/np.linalg.norm(CLV[n-1,:,:,count],axis=0,keepdims=True)
        
        if testzeroclv:
            tendcorr[n-1,count]=np.sum(np.multiply(CLV[n-1,:,minloc,count],tendency[n-1,:,count]))/np.sqrt(np.sum(np.multiply(tendency[n-1,:,count],tendency[n-1,:,count])))
            
        if tn % 1000 == 0:
            np.memmap.flush(R)
            np.memmap.flush(CLV)
            np.memmap.flush(lyaploc_clv)
            np.memmap.flush(tendcorr)
    lyapmean_clv[:,count]=np.mean(lyaploc_clv[int(tn/2):,:,count],axis=0)



print("Saveing results in folder "+savename+".")
np.save(savename+"/paraL96",paraL96)

invCLV = np.memmap(savename+'/invCLV.dat',mode='w+',shape=(len(t),dimN,M,1),dtype='float64')

for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
    invCLV[tn,:,:,0]=np.linalg.inv(CLV[tn,:,:,0])  

corrs=np.arange(0.01,1,0.1)
lowest=np.zeros((corrs.shape[0],M))
length=np.zeros((corrs.shape[0],M))
for n,c in enumerate(corrs):
    d = (np.logical_and(np.abs(tendcorr[:,0])>c,np.abs(tendcorr[:,0])<c+0.1))
    if d.any(): 
        length[n,:] = np.sum(d) 
        lowest[n,:] = np.average(lyaploc_clv[:,:,0],weights = d, axis = 0)
    else:
        lowest[n,:] = 0
maskcorr = (np.logical_and(np.abs(tendcorr[:,0])>0.95,np.abs(tendcorr[:,0])<1.01))