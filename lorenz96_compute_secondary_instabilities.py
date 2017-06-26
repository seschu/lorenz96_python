#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product


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
           'time' : np.arange(0,2000,0.1),
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
steplengthforsecondorder = np.arange(0,15,3)
hs=[1.0, 0.5] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
    
for paraL96,h in product([paraL96_2lay, paraL96_1lay],hs):
    if not paraL96['2lay'] and not h == 1.0: print("1 lay only with h = 1.");break
    savename=paraL96['expname']+"_h_"+str(h)
    spinup = paraL96['spinup']        
    paraL96=np.load(savename+"/paraL96.npy")
    paraL96=paraL96[()]
    # M number exponents
    if paraL96['2lay']:
        contract_func = lambda x,y: l96.contract_hess_l96_2layer_v2(x,y,paraL96)
        M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
        dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
    else:
        contract_func = l96.contract_hess_l96_1layer_v2
        M = paraL96['dimX'] 
        dimN = paraL96['dimX'] 
    
    np.save(savename+'/steplengthforsecondorder',steplengthforsecondorder)
    
    
    t = paraL96['time']
    CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M),dtype='float64')
    BLV = np.memmap(savename+'/BLV.dat',mode='r',shape=(len(t),dimN,M),dtype='float64')
    R = np.memmap(savename+'/R.dat',mode='r',shape=(len(t),dimN,M),dtype='float64')
    lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='r',shape=(M),dtype='float64')
    lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='r',shape=(M),dtype='float64')
    lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M),dtype='float64')
    lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='r',shape=(len(t)-1,M),dtype='float64')
    invCLV = np.memmap(savename+'/invCLV.dat',mode='r',shape=(len(t),dimN,M),dtype='float64')
    
    
    precision='float32'
    
    contracted_CLVs = np.memmap(savename+'/contracted_clvs.dat',mode='w+',shape=(len(t),dimN,M),dtype=precision) #(final_clv,init_clv)
    solution = np.memmap(savename+'/solution.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype=precision)
    full_solution = np.memmap(savename+'/full_solution.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype=precision)
    normalized_solution = np.memmap(savename+'/normalized_solution.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype=precision)
    growth = np.memmap(savename+'/growth.dat',mode='w+',shape=(len(steplengthforsecondorder),len(t),M),dtype=precision)
  
    
    for tn, (ts,te) in enumerate(zip(t[0:-20],t[1:-19])):
        dtau=te -ts
        print(tn)
        contracted_CLVs[tn,:,:]=1/2*contract_func(invCLV[tn ,:,:],CLV[tn,:,:])
        for n_step, len_step in enumerate(steplengthforsecondorder):
            # multiply with growth factor   
            for step in range(0,len_step+1):
                solution[n_step,tn,:,:] = solution[n_step,tn,:,:] + (
                dtau * np.multiply(np.exp(np.sum(dtau*lyaploc_clv[tn+step:tn+len_step,:,np.newaxis], axis =0)),np.multiply(contracted_CLVs[tn,:,:],np.exp(2*np.sum(dtau*lyaploc_clv[tn:tn+step,np.newaxis,:], axis =0))))
                )
            full_solution[n_step,tn,:,:] = np.matmul(CLV[tn,:,:],solution[n_step,tn,:,:])
            growth[n_step,tn,:]=np.linalg.norm(full_solution[n_step,tn,:,:],axis=0)
            normalized_solution[n_step,tn,:,:]=np.divide(solution[n_step,tn,:,:],np.linalg.norm(solution[n_step,tn,:,:],axis=0))
        if tn % 50 == 0:
            np.memmap.flush(solution)
            np.memmap.flush(contracted_CLVs)
            np.memmap.flush(normalized_solution)
            np.memmap.flush(full_solution)
            np.memmap.flush(growth)   
    print("Saveing results in folder "+savename+".")

