#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product


savename='secondaryinstabilities_v3'
paraL96=np.load(savename+"/paraL96.npy")
paraL96=paraL96[()]
# M number exponents
M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum

steplengthforsecondorder = [0, 1, 5, 10]


t = np.load(savename+'/t.npy')
CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64')
BLV = np.memmap(savename+'/BLV.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64')
R = np.memmap(savename+'/R.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='r',shape=(M,1),dtype='float64')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='r',shape=(M,1),dtype='float64')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M,1),dtype='float64')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='r',shape=(len(t)-1,M,1),dtype='float64')

#secondary_vector = np.memmap(savename+'/secondary_vector.dat',mode='w+',shape=(len(t),dimN,M,len(steplengthforsecondorder)),dtype='float64')
contracted_CLVs = np.memmap(savename+'/contracted_clvs.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype='float64') #(final_clv,init_clv)
solution = np.memmap(savename+'/solution.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype='float64')
normalized_solution = np.memmap(savename+'/solution.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype='float64')
growth = np.memmap(savename+'/growth.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),M),dtype='float64')
invCLV = np.memmap(savename+'/invCLV.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64')

#for tn, (ts,te) in enumerate(zip(t[0:-1],t[1:])):
#    invCLV[tn,:,:,0]=np.linalg.inv(CLV[tn,:,:,0])  
#    print(tn)

# Compute the exponents

for tn, (ts,te) in enumerate(zip(t[0:-20],t[1:-19])):
    dtau=te -ts
    print(tn)
    for n_step, len_step in enumerate(steplengthforsecondorder):
        print(tn,len_step )
        contracted_CLVs[n_step,tn,:,:]=1/2*l96.contract_hess_l96_2layer_v2(invCLV[tn+len_step ,:,:,0],CLV[tn,:,:,0],paraL96)
        # multiply with growth factor   
        for step in range(0,n_step+1):
            solution[n_step,tn,:,:] = solution[n_step,tn,:,:] + (
            dtau * np.multiply(np.exp(np.sum(dtau*lyaploc_clv[tn+step:tn+len_step,:,np.newaxis,0], axis =0)),np.multiply(contracted_CLVs[n_step,tn,:,:],np.exp(2*np.sum(dtau*lyaploc_clv[tn:tn+step,np.newaxis,:,0], axis =0))))
            )
        growth[n_step,tn,:]=np.linalg.norm(np.matmul(CLV[tn+len_step ,:,:,0],solution[n_step, tn,:,:]),axis = 0)
        growth[n_step,tn,:]=np.divide(growth[n_step,tn,:],growth[0,tn,:])        
        normalized_solution[n_step,tn,:,:]=np.divide(solution[n_step,tn,:,:],growth[n_step,tn,np.newaxis,:])
    if tn % 10 == 0:
        np.memmap.flush(solution)
        np.memmap.flush(contracted_CLVs)
        np.memmap.flush(normalized_solution)
        
print("Saveing results in folder "+savename+".")
