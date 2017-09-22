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
           'time' : np.arange(0,500,0.1),
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
           'time' : np.arange(0,500,0.1),
           'spinup' : 100,
           '2lay' : False
           }


testzeroclv=True
steplengthforsecondorder = np.arange(0,200,1)
hs=[1.0]#, 0.5] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
precision='float64'    
for paraL96,h in product([paraL96_1lay],hs):
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
    CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M),dtype='float64', order = 'F')
    lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M),dtype='float64', order = 'F')
    invCLV = np.memmap(savename+'/invCLV.dat',mode='r',shape=(len(t),dimN,M),dtype='float64', order = 'F')
    
    
    
    lensteparray = len(steplengthforsecondorder)
    contracted_CLVs = np.memmap(savename+'/contracted_clvs.dat',mode='w+',shape=(len(t),dimN,M),dtype=precision, order = 'F') #(final_clv,init_clv)
    solution = np.memmap(savename+'/solution.dat',mode='w+',shape=(len(t),lensteparray,dimN,M),dtype=precision, order = 'F')
    full_solution = np.memmap(savename+'/full_solution.dat',mode='w+',shape=(len(t),lensteparray,dimN,M),dtype=precision, order = 'F')
    #normalized_solution = np.memmap(savename+'/normalized_solution.dat',mode='w+',shape=(len(t),lensteparray,dimN,M),dtype=precision, order = 'F')
    growth = np.memmap(savename+'/growth.dat',mode='w+',shape=(len(t),lensteparray,M),dtype=precision, order = 'F')
    for tn, (ts,te) in enumerate(zip(t[0:-2],t[1:-1])):
        print(tn)
        contracted_CLVs[tn,:,:]=1/2*contract_func(invCLV[tn,:,:],CLV[tn,:,:])
    
    np.memmap.flush(contracted_CLVs)

    
    for tn, (ts,te) in enumerate(zip(t[0:-steplengthforsecondorder.max()-1],t[1:-steplengthforsecondorder.max()])):
        dtau=te -ts
        print(tn)
        for n_ind, n_step in enumerate(steplengthforsecondorder):
            if n_ind == 0: solution[tn,n_ind,:,:] = 0
            else:
                tni = tn 
                solution[tn,n_ind,:,:] = (solution[tn,n_ind-1,:,:] + 
            dtau * np.multiply(np.exp(-dtau*np.memmap.sum(lyaploc_clv[tni:tn+n_step,:,np.newaxis], axis =0)),np.multiply(contracted_CLVs[tn+n_step,:,:],np.exp(2.0*np.memmap.sum(dtau*lyaploc_clv[tni:tn+n_step,np.newaxis,:], axis =0))))
            )
        for n_ind, n_step in enumerate(steplengthforsecondorder):
            solution[tn,n_ind,:,:] = np.multiply(solution[tn,n_ind,:,:] ,np.exp(np.sum(dtau*lyaploc_clv[tn:tn+n_step,:,np.newaxis], axis = 0)))
            if n_ind == 0: continue
            else:
                full_solution[tn,n_ind,:,:] = np.matmul(CLV[tn+n_step,:,:],solution[tn,n_step,:,:])
                growth[tn,n_ind,:]=np.log(np.linalg.norm(full_solution[tn,n_ind,:,:],axis=0))/(n_step*dtau)

        if tn % 1 == 0:
            np.memmap.flush(solution)
            np.memmap.flush(contracted_CLVs)
            #np.memmap.flush(normalized_solution)
            np.memmap.flush(full_solution)
            np.memmap.flush(growth)   
    print("Saveing results in folder "+savename+".")

