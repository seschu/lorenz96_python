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


testzeroclv=True

savename='secondaryinstabilities'
CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV = np.memmap(savename+'/BLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R = np.memmap(savename+'/R.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='r',shape=(len(t)-1,M,len(hs)),dtype='float64')

secondary_vector = np.memmap(savename+'/secondary_vector.dat',mode='w+',shape=(len(t),dimN,M,len(hs)),dtype='float64')

# Compute the exponents

n_step = 1

propagator = np.eye(dimN,M)

for count,h in enumerate(hs):
    paraL96['h']=h
    
    for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
        # create propagator
        propagator=np.matmul(np.matmul(BLV[tn,:,:,count],R[tn,:,:,count]),propagator)
        
        # mutiply clvs and contract with hessematrix
        
        # multiply with growth factor
        
        if tn % n_step == 0: 
            # propagate solution 
            
            # reset propagator
            propagator = np.eye(dimN,M) 

print("Saveing results in folder "+savename+".")
