#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg

# these are our constants
paraL96 = {'F1' : 6,
           'F2' : 6,
           'b'  : 10,
           'c'  : 10,
           'h'  : 1,
           'dimX': 36,
           'dimY' : 0,
           'RescaledY' : False
           }

# M number exponents
M = paraL96['dimX'] #+ paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] #+ paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
integrator = 'rk4'
t = np.arange(0,200,0.005)
spinup = 100;
#setup L96
hs=[ 0. ] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]


savename='gabrielessetup_convergence_dimY_0'
if not os.path.exists(savename): os.mkdir(savename)
CLV = np.memmap(savename+'/CLV.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV = np.memmap(savename+'/BLV.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R = np.memmap(savename+'/R.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='w+',shape=(M,len(hs)),dtype='float64')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='w+',shape=(M,len(hs)),dtype='float64')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='w+',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='w+',shape=(len(t)-1,M,len(hs)),dtype='float64')

CLV2 = np.memmap(savename+'/CLV2.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV2 = np.memmap(savename+'/BLV2.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R2 = np.memmap(savename+'/R2.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv2 = np.memmap(savename+'/lyapmean_blv2.dat',mode='w+',shape=(M,len(hs)),dtype='float64')
lyapmean_clv2 = np.memmap(savename+'/lyapmean_clv2.dat',mode='w+',shape=(M,len(hs)),dtype='float64')
lyaploc_clv2 = np.memmap(savename+'/lyaploc_clv2.dat',mode='w+',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv2 = np.memmap(savename+'/lyaploc_blv2.dat',mode='w+',shape=(len(t)-1,M,len(hs)),dtype='float64')

correlationclv = np.memmap(savename+'/correlationclv.dat',mode='w+',shape=(len(t),M,len(hs)),dtype='float64')


# Compute the exponents

for count,h in enumerate(hs):
    paraL96['h']=h
    print("\nExperiment is the following:")
    for key in paraL96.keys(): print(key+' : '+str(paraL96[key]))
    L96,L96Jac,L96JacV,L96JacFull,dimN = l96.setupL96(paraL96)
    field = l96.GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=integrator)
    field2 = l96.GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=integrator)
    # initialize fields 
    print("\nInitialize ...")
    field.init_back('random',0.1)
    field.init_lin('random',0.1)
    field.restoreQ()   
    # spinup
    print("\nSpinup ...")
    #for i in range(0,int(spinup/0.1),1): 
    field.integrate_back(spinup,dt=np.mean(np.diff(t)))
    field.step_t = 0.0
    # save initial state
    field2.init_back(field.x['back'])
    field2.step_t = 0.0
    
    # first run
    BLV[0,:,:,count]=field.x['lin']
    print("\nQR Steps ...")
    # Do QR step
    fieldstore=np.zeros((len(t),dimN,2),dtype=np.float64)
    for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
        fieldstore[tn,:,0]=field.x['back']
        field.qr_step(te-ts,dt=np.mean(np.diff(t)))
        R[tn,:,:,count]=field.R
        BLV[tn+1,:,:,count]=field.x['lin']
        #condnb[tn+1,count]=np.linalg.cond(BLV[tn+1,:,:,count],p=2)/np.linalg.cond(BLV[tn+1,:,:,count],p=-2)
        #print(np.linalg.cond(BLV[tn+1,:,:,count],p=2)/np.linalg.cond(BLV[tn+1,:,:,count],p=-2))
        print(te,'|',count+1,'/',len(hs))
        lyaploc_blv[tn,:,count]=field.lyap
        if tn % 10 == 0:
            np.memmap.flush(BLV)
            np.memmap.flush(R)
            np.memmap.flush(lyaploc_blv)
    lyapmean_blv[:,count]=np.mean(lyaploc_blv[int(tn/2):,:,count],axis=0)
    
    # Do Backwards steps

    print("\nBackwards Steps ...")
    imax = tn
    res=triu(np.random.rand(M,M))
    CLV[imax,:,:,count]=np.matmul(BLV[imax,:,:,count],res)
    res=res/np.linalg.norm(res,axis=0,keepdims=True)
    lyaploc_clv[imax,:,count]=np.log(1/np.abs(np.linalg.norm(CLV[imax,:,:,count],axis=0)))/np.abs(te-ts)
        
    for tn, (ts,te) in enumerate(zip(t[-2:0:-1],t[-1:0:-1])):
        n = imax-tn
        if n==0: break
        res=linalg.solve(R[n-1,0:M,:,count],res)
        CLV[n-1,:,:,count]=np.matmul(BLV[n-1,:,:,count],res)
        res=res/np.linalg.norm(res,axis=0,keepdims=True)
        lyaploc_clv[n-1,:,count]=np.log(1/np.abs(np.linalg.norm(CLV[n-1,:,:,count],axis=0)))/(te-ts)
        CLV[n-1,:,:,count]= CLV[n-1,:,:,count]/np.linalg.norm(CLV[n-1,:,:,count],axis=0,keepdims=True)
        if tn % 1000 == 0:
            np.memmap.flush(R)
            np.memmap.flush(CLV)
            np.memmap.flush(lyaploc_clv)
    lyapmean_clv[:,count]=np.mean(lyaploc_clv[int(tn/2):,:,count],axis=0)



print("Saveing results in folder "+savename+".")


for count,h in enumerate(hs):        
    # second run
    print("\nQR Steps ...")
    # Do QR step
    
    field2.init_lin('random',0.1)
    field2.step_t = 0.0
    field2.restoreQ()   
    BLV2[0,:,:,count]=field2.x['lin']
    
    for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
        fieldstore[tn,:,1]=field2.x['back']
        field2.qr_step(te-ts,dt=np.mean(np.diff(t)))
        R2[tn,:,:,count]=field2.R
        BLV2[tn+1,:,:,count]=field2.x['lin']
        #condnb[tn+1,count]=np.linalg.cond(BLV[tn+1,:,:,count],p=2)/np.linalg.cond(BLV[tn+1,:,:,count],p=-2)
        #print(np.linalg.cond(BLV[tn+1,:,:,count],p=2)/np.linalg.cond(BLV[tn+1,:,:,count],p=-2))
        print(te,'|',count+1,'/',len(hs))
        lyaploc_blv2[tn,:,count]=field2.lyap
        if tn % 1000 == 0:
            np.memmap.flush(BLV2)
            np.memmap.flush(R2)
            np.memmap.flush(lyaploc_blv2)
    lyapmean_blv2[:,count]=np.mean(lyaploc_blv2[int(tn/2):,:,count],axis=0)
    
    # Do Backwards steps

    print("\nBackwards Steps ...")
    imax = tn
    res=triu(np.random.rand(M,M))
    CLV2[imax,:,:,count]=np.matmul(BLV2[imax,:,:,count],res)
    res=res/np.linalg.norm(res,axis=0,keepdims=True)
    lyaploc_clv2[imax,:,count]=np.log(1/np.abs(np.linalg.norm(CLV2[imax,:,:,count],axis=0)))/np.abs(te-ts)
        
    for tn, (ts,te) in enumerate(zip(t[-2:0:-1],t[-1:0:-1])):
        n = imax-tn
        if n==0: break
        res=linalg.solve(R2[n-1,0:M,:,count],res)
        CLV2[n-1,:,:,count]=np.matmul(BLV2[n-1,:,:,count],res)
        res=res/np.linalg.norm(res,axis=0,keepdims=True)
        lyaploc_clv2[n-1,:,count]=np.log(1/np.abs(np.linalg.norm(CLV2[n-1,:,:,count],axis=0)))/(te-ts)
        CLV2[n-1,:,:,count]= CLV2[n-1,:,:,count]/np.linalg.norm(CLV2[n-1,:,:,count],axis=0,keepdims=True)
        if tn % 1000 == 0:
            np.memmap.flush(CLV2)
            np.memmap.flush(lyaploc_clv2)
    lyapmean_clv2[:,count]=np.mean(lyaploc_clv2[int(tn/2):,:,count],axis=0)
        
for clv in range(0,dimN):
    print(clv)
    correlationclv[1:-2,clv,0] = np.sum(np.abs(np.multiply(CLV[1:-2,:,clv,0],CLV2[1:-2,:,clv,0])),axis=1)
    np.memmap.flush(correlationclv)

projX = np.memmap(savename+'/projX.dat',mode='w+',shape=(len(t),M,len(hs)),dtype='float64')
projX2 = np.memmap(savename+'/projX2.dat',mode='w+',shape=(len(t),M,len(hs)),dtype='float64')

for clv in range(0,M):
    print(clv)
    projX[0:-1,clv,0]=np.sum(CLV[0:-1,0:paraL96['dimX'],clv,0]**2,axis=1)
    projX2[0:-1,clv,0]=np.sum(CLV2[0:-1,0:paraL96['dimX'],clv,0]**2,axis=1)
    