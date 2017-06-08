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


integrator='classic'

dt=10**-5

# M number exponents
M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum

t = np.arange(0,50,0.01)
spinup = 50;
#setup L96
hs=[0.5]
CLV = np.zeros((len(t),dimN,M))
BLV = np.zeros((len(t),dimN,M))
R = np.zeros((len(t),dimN,M))
lyapmean_blv = np.zeros((M))
lyapmean_clv = np.zeros((M))
lyaploc_clv = np.zeros((len(t),M))
lyaploc_blv = np.zeros((len(t)-1,M))
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
field.integrate_back(spinup,integrator=integrator,dt= dt)
lyap = np.zeros(M)
BLV[0,:,:]=field.x['lin']
field.step_t = 0.0
print("\nQR Steps ...")
# Do QR step

for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
    field.qr_step(te-ts,integrator=integrator,dt= dt)
    R[tn,:,:]=field.R
    BLV[tn+1,:,:]=field.x['lin']
    #condnb[tn+1]=np.linalg.cond(BLV[tn+1,:,:],p=2)/np.linalg.cond(BLV[tn+1,:,:],p=-2)
    #print(np.linalg.cond(BLV[tn+1,:,:],p=2)/np.linalg.cond(BLV[tn+1,:,:],p=-2))
    print(te)
    lyaploc_blv[tn,:]=field.lyap
lyapmean_blv[:]=np.mean(lyaploc_blv[int(tn/2):,:],axis=0)

# Do Backwards steps

print("\nBackwards Steps ...")
imax = tn
res=triu(np.random.rand(M,M))
CLV[imax,:,:]=np.matmul(BLV[imax,:,:],res)
res=res/np.linalg.norm(res,axis=0,keepdims=True)
lyaploc_clv[imax,:]=np.log(1/np.abs(np.linalg.norm(CLV[imax,:,:],axis=0)))/np.abs(te-ts)
    
for tn, (ts,te) in enumerate(zip(t[-2:0:-1],t[-1:0:-1])):
    n = imax-tn
    if n==0: break
    res=linalg.solve(R[n-1,0:M,:],res)
    CLV[n-1,:,:]=np.matmul(BLV[n-1,:,:],res)
    res=res/np.linalg.norm(res,axis=0,keepdims=True)
    lyaploc_clv[n-1,:]=np.log(1/np.abs(np.linalg.norm(CLV[n-1,:,:],axis=0)))/(te-ts)
    CLV[n-1,:,:]= CLV[n-1,:,:]/np.linalg.norm(CLV[n-1,:,:],axis=0,keepdims=True)
lyapmean_clv[:]=np.mean(lyaploc_clv[int(tn/2):,:],axis=0)
            


savename='simple_run'
print("Saveing results in folder "+savename+".")

if not os.path.exists(savename): os.mkdir(savename)
np.save(savename+'/lyaploc_blv',lyaploc_blv)
np.save(savename+'/lyaploc_clv',lyaploc_clv)
np.save(savename+'/lyapmean_blv',lyapmean_blv)
np.save(savename+'/lyapmean_clv',lyapmean_clv)
np.save(savename+'/paraL96',paraL96)
np.save(savename+'/M',M)
np.save(savename+'/imax',imax)
np.save(savename+'/h',hs)
np.save(savename+'/CLV',CLV)
np.save(savename+'/BLV',BLV)
np.save(savename+'/R',R)
