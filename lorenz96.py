#!/home/user/anaconda3/bin/python
import numpy as np
from l96 import *

# these are our constants
paraL96 = {'F1' : 8,
           'F2' : 0,
           'b'  : 10,
           'c'  : 10,
           'h'  : 1,
           'dimX': 36,
           'dimY' : 8,
           'RescaledY' : False
           }

# M number exponents
M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum

t = np.arange(0,50,1)
spinup = 100;
#setup L96
hs=[ 0.    ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
CLV = np.zeros((len(t),dimN,M,len(hs),2))
BLV = np.zeros((len(t),dimN,M,len(hs),2))
R = np.zeros((len(t),dimN,M,len(hs),2))
lyapmean_blv = np.zeros((M,len(hs),2))
lyapmean_clv = np.zeros((M,len(hs),2))
lyaploc_clv = np.zeros((len(t),M,len(hs),2))
lyaploc_blv = np.zeros((len(t)-1,M,len(hs),2))

condnb = np.zeros((len(t),len(hs),2))



for counti,rescale in enumerate([False, True]):
    paraL96['RescaledY']=rescale
    # Compute the exponents

    for count,h in enumerate(hs):
        paraL96['h']=h
        print("\nExperiment is the following:")
        for key in paraL96.keys(): print(key+' : '+str(paraL96[key]))
        L96,L96Jac,L96JacV,L96JacFull,dimN = setupL96_2layer(paraL96)
        field = GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull)
        # initialize fields 
        print("\nInitialize ...")
        field.init_back('random',0.1)
        field.init_lin('random',0.1)
        field.restoreQ()   
        # spinup
        print("\nSpinup ...")
        for i in range(0,int(spinup/0.1),1): field.integrate_back(0.1)
        lyap = np.zeros(M)
        BLV[0,:,:,count,counti]=field.x['lin']
        field.step_t = 0.0
        print("\nQR Steps ...")
        # Do QR step
        
        for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
            field.qr_step(te-ts)
            R[tn,:,:,count,counti]=field.R
            BLV[tn+1,:,:,count,counti]=field.x['lin']
            #condnb[tn+1,count,counti]=np.linalg.cond(BLV[tn+1,:,:,count,counti],p=2)/np.linalg.cond(BLV[tn+1,:,:,count,counti],p=-2)
            #print(np.linalg.cond(BLV[tn+1,:,:,count,counti],p=2)/np.linalg.cond(BLV[tn+1,:,:,count,counti],p=-2))
            print(te)
            lyaploc_blv[tn,:,count,counti]=field.lyap
        lyapmean_blv[:,count,counti]=np.mean(lyaploc_blv[int(tn/2):,:,count,counti],axis=0)
        
        # Do Backwards steps

        print("\nBackwards Steps ...")
        imax = tn
        res=triu(np.random.rand(M,M))
        CLV[imax,:,:,count,counti]=np.matmul(BLV[imax,:,:,count,counti],res)
        res=res/np.linalg.norm(res,axis=0,keepdims=True)
        lyaploc_clv[imax,:,count,counti]=np.log(1/np.abs(np.linalg.norm(CLV[imax,:,:,count,counti],axis=0)))/np.abs(te-ts)
            
        for tn, (ts,te) in enumerate(zip(t[-2:0:-1],t[-1:0:-1])):
            n = imax-tn
            if n==0: break
            res=linalg.solve(R[n-1,0:M,:,count,counti],res)
            CLV[n-1,:,:,count,counti]=np.matmul(BLV[n-1,:,:,count,counti],res)
            res=res/np.linalg.norm(res,axis=0,keepdims=True)
            lyaploc_clv[n-1,:,count,counti]=np.log(1/np.abs(np.linalg.norm(CLV[n-1,:,:,count,counti],axis=0)))/(te-ts)
            CLV[n-1,:,:,count,counti]= CLV[n-1,:,:,count,counti]/np.linalg.norm(CLV[n-1,:,:,count,counti],axis=0,keepdims=True)
        lyapmean_clv[:,count,counti]=np.mean(lyaploc_clv[int(tn/2):,:,count,counti],axis=0)
            


savename='traditionalrun'
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
