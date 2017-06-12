#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product

# these are our constants
paraL96 = {'F1' : 10,
           'F2' : 0,
           'b'  : 10,
           'c'  : 10,
           'h'  : 0,
           'dimX': 36,
           'dimY' : 10,
           'RescaledY' : False
           }

# M number exponents
M = paraL96['dimX'] #+ paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX']# + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
integrator = 'classic'
t = np.arange(0,10000,1)
spinup = 100;
#setup L96
hs=[ 1. ] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]


savename='secondaryinstabilities'
CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV = np.memmap(savename+'/BLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R = np.memmap(savename+'/R.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='r',shape=(len(t)-1,M,len(hs)),dtype='float64')

secondary_vector = np.memmap(savename+'/secondary_vector.dat',mode='w+',shape=(len(t),dimN,M,15),dtype='float64')
contracted_CLVs = np.memmap(savename+'/contracted_clvs.dat',mode='w+',shape=(len(t),dimN,M,15),dtype='float64')
solution = np.memmap(savename+'/solution.dat',mode='w+',shape=(len(t),dimN,M,15),dtype='float64')
normalized_solution = np.memmap(savename+'/solution.dat',mode='w+',shape=(len(t),dimN,M,15),dtype='float64')
growth = np.memmap(savename+'/growth.dat',mode='w+',shape=(len(t),dimN,M,15),dtype='float64')

# Compute the exponents

n_step = 1

t = np.arange(0,10000,1)


for tn, (ts,te) in enumerate(zip(t[0:-20],t[1:-19])):
    dtau=te -ts
    print(tn)
    for n_step in range(0,15):
        print(tn,n_step)
        invertCLV_final=np.linalg.inv(CLV[tn+n_step,:,:,0])
        for final_clv,init_clv in product(np.arange(0,M),np.arange(0,M)):
            contracted_CLVs[tn,init_clv,final_clv,n_step]=1/2*l96.contract_hess_l96_1layer_v2(invertCLV_final[final_clv,:],CLV[tn,:,init_clv,0])
            # multiply with growth factor   
            for step in range(0,n_step+1):
                solution[tn,init_clv,final_clv,n_step] = solution[tn,init_clv,final_clv,n_step] + (
                dtau * contracted_CLVs[tn,init_clv,final_clv,n_step]*np.exp(2*np.sum(dtau*lyaploc_clv[tn:tn+step,final_clv,0]))*np.exp(np.sum(dtau*lyaploc_clv[tn+step:tn+n_step,init_clv,0]))
                )
        #growth[tn,init_clv,n_step]=np.linalg.norm(np.matmul(CLV[tn+n_step,:,:,0],solution[tn,init_clv,:,n_step]))
        #growth[tn,init_clv,n_step]=np.divide(growth[tn,init_clv,n_step],growth[tn,init_clv,0])        
        #normalized_solution[tn,init_clv,:,n_step]=np.divide(solution[tn,init_clv,:,n_step],np.linalg.norm(np.matmul(CLV[tn+n_step,:,:,0],solution[tn,init_clv,:,n_step])))
    if tn % 100 == 0:
        np.memmap.flush(solution)
        np.memmap.flush(contracted_CLVs)
        
print("Saveing results in folder "+savename+".")
