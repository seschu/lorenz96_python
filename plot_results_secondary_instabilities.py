#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product
import matplotlib.pyplot as plt

resultsfolder='secondaryinstabilities'



CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64', order = 'F')
BLV = np.memmap(savename+'/BLV.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64', order = 'F')
R = np.memmap(savename+'/R.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64', order = 'F')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='r',shape=(M,1),dtype='float64', order = 'F')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='r',shape=(M,1),dtype='float64', order = 'F')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M,1),dtype='float64', order = 'F')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='r',shape=(len(t)-1,M,1),dtype='float64', order = 'F')

precision='float32'
contracted_CLVs = np.memmap(savename+'/contracted_clvs.dat',mode='r',shape=(len(t),dimN,M),dtype='float64', order = 'F') #(final_clv,init_clv)
solution = np.memmap(savename+'/solution.dat',mode='r',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype='float64', order = 'F')
normalized_solution = np.memmap(savename+'/solution.dat',mode='r',shape=(len(steplengthforsecondorder),len(t),dimN,M),dtype='float64', order = 'F')
growth = np.memmap(savename+'/growth.dat',mode='r',shape=(len(steplengthforsecondorder),len(t),M),dtype='float64', order = 'F')
invCLV = np.memmap(savename+'/invCLV.dat',mode='r',shape=(len(t),dimN,M,1),dtype='float64', order = 'F')


    
A=np.zeros((dimN,dimN,len(steplengthforsecondorder)))
for n_step, len_step in enumerate(steplengthforsecondorder):
    dummy=np.median(np.abs(solution[n_step,maskcorr,:,:]),axis=0)
    A[:,:,n_step]=np.divide(dummy,np.linalg.norm(dummy,axis=0))

minloc=np.argmin(np.abs(lyaploc_clv[maskcorr,]))

plt.figure()
plt.contourf(A[:,:,2])
plt.xlabel('Linear Perturbation')
plt.ylabel('Second Order Projection onto CLVs')           
plt.colorbar()
plt.title('Projection of 2nd order onto CLVs (y axis) if linear perturbation along certain CLV (x axis)')