#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product
import matplotlib.pyplot as plt

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
           'time' : np.arange(0,1000,0.1),
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


resultsfolder = 'secondaryinstabilities'
if not os.path.exists(resultsfolder): os.mkdir(resultsfolder)
experiments = [paraL96_2lay]

testzeroclv=True
steplengthforsecondorder = np.arange(0,15,3)
hs=[1.0] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]

compute = True #compute array projections

averageintervall=np.arange(4000,8500)

for paraL96,h in product(experiments,hs):
    #print(paraL96,h)
    if not paraL96['2lay'] and h == 0.5: break
    savename=paraL96['expname']+"_h_"+str(h)
    spinup = paraL96['spinup']        
    paraL96=np.load(savename+"/paraL96.npy")
    paraL96=paraL96[()]
    steplengthforsecondorder = np.load(savename+'/steplengthforsecondorder.npy')
    # M number exponents
    if paraL96['2lay']:
        M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
        dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
    else:
        M = paraL96['dimX'] 
        dimN = paraL96['dimX'] 
    maskcorr=np.load(savename+"/maskcorr.npy")
    
    #CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(paraL96['time']),dimN,M,1),dtype='float64')
    #lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(paraL96['time']),M,1),dtype='float64')
    precision='float32'
    if compute: solution = np.memmap(savename+'/solution.dat',mode='r',shape=(len(steplengthforsecondorder),len(paraL96['time']),dimN,M),dtype=precision)
    #full_solution = np.memmap(savename+'/full_solution.dat',mode='r',shape=(len(steplengthforsecondorder),len(paraL96['time']),dimN,M),dtype=precision)
    #normalized_solution = np.memmap(savename+'/normalized_solution.dat',mode='r',shape=(len(steplengthforsecondorder),len(paraL96['time']),dimN,M),dtype=precision)
    #growth = np.memmap(savename+'/growth.dat',mode='r',shape=(len(steplengthforsecondorder),len(paraL96['time']),M),dtype=precision)
    if compute:  projections=np.zeros((len(steplengthforsecondorder),dimN,M))
    else: projections = np.load(savename+"/projections.npy")
    if compute:
        #for step, len_step in enumerate(steplengthforsecondorder):
        #print(step)
        dummy = np.zeros((len(steplengthforsecondorder),solution.shape[2],solution.shape[3]))
        for tn in averageintervall[np.arange(0,averageintervall.shape[0],10)]:
            print(tn)
            dummy = dummy + np.sum(np.abs(solution[:,tn:tn+10,:,:]),axis=1).copy()
            #np.memmap.flush(solution)
        projections[:,:,:]=np.divide(dummy,np.linalg.norm(dummy,axis=1,keepdims=True))
    for step, len_step in enumerate(steplengthforsecondorder):
        fig, axarr = plt.subplots(2, figsize = (7,6))
        X,Y = np.meshgrid(range(0,dimN),range(0,dimN))
        im=axarr[0].pcolormesh(X,Y,projections[step,:,:])
        axarr[0].set_xlabel('Linear Perturbation',fontsize = 8)
        axarr[0].set_ylabel('Second Order Projection onto CLVs',fontsize = 8)           
        fig.colorbar(im, ax=axarr[0])
        dt=np.mean(np.diff(paraL96['time']))
        axarr[0].set_title('Projection onto CLVs (y axis) if linear perturbation\n along CLV (x axis), Delay '+r'$\tau$\ ='+str(steplengthforsecondorder[step]*dt)+' MTU',
                  fontsize = 8)
        X,Y = np.meshgrid(range(1,61),range(1,61))
        im2=axarr[1].pcolormesh(X,Y,projections[step,0:60,0:60])
        axarr[1].set_xlabel('Linear Perturbation',fontsize = 8)
        axarr[1].set_ylabel('Second Order Projection onto CLVs',fontsize = 8)           
        fig.colorbar(im2, ax=axarr[1])
        dt=np.mean(np.diff(paraL96['time']))
        axarr[1].set_title('Projection onto CLVs (y axis) if linear perturbation\n along CLV (x axis), Delay '+r'$\tau$\ ='+str(steplengthforsecondorder[step]*dt)+' MTU',
                  fontsize = 8)
        
        fig.tight_layout()
        figname = "2_lay_projections_h_"+str(h)+"_step_"+str(step) if paraL96['2lay'] else "1_lay_projections_step_"+str(step)
        fig.savefig(resultsfolder+"/"+figname+".pdf")
        fig.savefig(resultsfolder+"/"+figname+".png")
    if compute: np.save(savename+"/projections",projections)