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
           'time' : np.arange(0,100,0.1),
           'spinup' : 100,
           '2lay' : False
           }


resultsfolder = 'secondaryinstabilities'
if not os.path.exists(resultsfolder): os.mkdir(resultsfolder)
for fold in ['projections', 'correlations']:
        if not os.path.exists(resultsfolder+"/"+fold): os.mkdir(resultsfolder+"/"+fold)

experiments = [paraL96_1lay]

precision='float64'

testzeroclv=False

hs=[1.0] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]

compute = True #compute array projections

averageintervall=np.arange(1000,4000)

for paraL96,h in product(experiments,hs):
    
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
    
    CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(paraL96['time']),dimN,M),dtype='float64')
    
    if compute: 
        solution = np.memmap(savename+'/solution.dat',mode='r',shape=(len(paraL96['time']),len(steplengthforsecondorder),dimN,M),dtype=precision,order = 'F')
        full_solution = np.memmap(savename+'/full_solution.dat',mode='r',shape=(len(paraL96['time']),len(steplengthforsecondorder),dimN,M),dtype=precision, order = 'F')
    if compute:  
        projections=np.zeros((len(steplengthforsecondorder),dimN,M))
        correlations=np.zeros((len(steplengthforsecondorder),dimN,M))
    else: 
        projections = np.load(savename+"/projections.npy")
        correlations = np.load(savename+"/correlations.npy")
    if compute:
        dummy = np.zeros((len(steplengthforsecondorder),solution.shape[2],solution.shape[3]))
#        for tn in averageintervall:
#            print(tn)
#            dummy = dummy + np.abs(solution[tn,:,:,:])
#        dummy = dummy/len(averageintervall)
        dummy = np.memmap.mean(np.abs(solution[averageintervall,:,:,:]))
        projections[1:,:,:]=np.divide(dummy[1:,:,:], np.linalg.norm(dummy[1:,:,:], axis = 1, keepdims=True))
        
        
        dummy = np.zeros((len(steplengthforsecondorder),solution.shape[2],solution.shape[3]))
        
        
        for tn in averageintervall:
            for ist in range(0,len(steplengthforsecondorder)):
                if not ist == 0:
                    normalized=np.divide(full_solution[tn,ist,:,:], np.linalg.norm(full_solution[tn,ist,:,:], axis = 0, keepdims=True))               
                    dummy[ist,:,:] = dummy[ist,:,:] + np.abs(np.matmul(np.transpose(CLV[tn+ist,:,:]), normalized))
                else: dummy[ist,:,:]=0
        dummy = dummy/len(averageintervall)    
        correlations[1:,:,:]=dummy[1:,:,:]

    for step, len_step in enumerate(steplengthforsecondorder):
        print(step,len_step)
        
        # print part depending on 2layer or 1layer setup
        
        # Plot Projections on CLVs
        
        if paraL96['2lay']: fig, axarr = plt.subplots(2, figsize = (7,6))
        else: fig = plt.figure(); axarr=[0];axarr[0]=plt.gca()
        X,Y = np.meshgrid(range(1,dimN+1),range(1,dimN+1))
        im=axarr[0].contourf(X,Y,projections[step,:,:])
        axarr[0].set_xlabel('Linear Perturbation',fontsize = 8)
        axarr[0].set_ylabel('Second Order Projection onto CLVs',fontsize = 8)           
        fig.colorbar(im, ax=axarr[0])
        dt=np.mean(np.diff(paraL96['time']))
        axarr[0].set_title('Projection onto CLVs (y axis) if linear perturbation\n along CLV (x axis), Delay '+r'$\tau$ ='+str(steplengthforsecondorder[step]*dt)+' MTU',
                  fontsize = 8)
        if paraL96['2lay']: 
            X,Y = np.meshgrid(range(1,np.min((61,dimN+1))),range(1,np.min((61,dimN+1))))
            im2=axarr[1].contourf(X,Y,projections[step,0:np.min((60,dimN)),0:np.min((60,dimN))])
            axarr[1].set_xlabel('Linear Perturbation',fontsize = 8)
            axarr[1].set_ylabel('Second Order Projection onto CLVs',fontsize = 8)           
            fig.colorbar(im2, ax=axarr[1])
            dt=np.mean(np.diff(paraL96['time']))
            axarr[1].set_title('Projection onto CLVs (y axis) if linear perturbation\n along CLV (x axis), Delay '+r'$\tau$ ='+str(steplengthforsecondorder[step]*dt)+' MTU',
                      fontsize = 8)
        
        fig.tight_layout()
        figname = "2_lay_projections_h_"+str(h)+"_step_"+str(step) if paraL96['2lay'] else "1_lay_projections_step_"+str(step)
        fig.savefig(resultsfolder+"/projections/"+figname+".pdf")
        fig.savefig(resultsfolder+"/projections/"+figname+".png", dpi =400)
        plt.close(fig)
        
        
        
        # Plot Correlations with CLVs
        
        if paraL96['2lay']: fig, axarr = plt.subplots(2, figsize = (7,6))
        else: fig = plt.figure(); axarr=[0];axarr[0]=plt.gca()
        X,Y = np.meshgrid(range(1,dimN+1),range(1,dimN+1))
        im=axarr[0].contourf(X,Y,correlations[step,:,:])
        axarr[0].set_xlabel('Linear Perturbation',fontsize = 8)
        axarr[0].set_ylabel('Correlation Projection with CLVs',fontsize = 8)           
        fig.colorbar(im, ax=axarr[0])
        dt=np.mean(np.diff(paraL96['time']))
        axarr[0].set_title('Correlation with CLVs (y axis) if linear perturbation\n along CLV (x axis), Delay '+r'$\tau$ ='+str(steplengthforsecondorder[step]*dt)+' MTU',
                  fontsize = 8)
        if paraL96['2lay']: 
            X,Y = np.meshgrid(range(1,np.min((61,dimN+1))),range(1,np.min((61,dimN+1))))
            im2=axarr[1].contourf(X,Y,correlations[step,0:np.min((60,dimN)),0:np.min((60,dimN))])
            axarr[1].set_xlabel('Linear Perturbation',fontsize = 8)
            axarr[1].set_ylabel('Correlation of Second Order withCLVs',fontsize = 8)           
            fig.colorbar(im2, ax=axarr[1])
            dt=np.mean(np.diff(paraL96['time']))
            axarr[1].set_title('Correlation with CLVs (y axis) if linear perturbation\n along CLV (x axis), Delay '+r'$\tau$ ='+str(steplengthforsecondorder[step]*dt)+' MTU',
                      fontsize = 8)
        
        fig.tight_layout()
        figname = "2_lay_correlations_h_"+str(h)+"_step_"+str(step) if paraL96['2lay'] else "1_lay_correlations_step_"+str(step)
        fig.savefig(resultsfolder+"/correlations/"+figname+".pdf")
        fig.savefig(resultsfolder+"/correlations/"+figname+".png", dpi =400)
        plt.close(fig)
    if compute: 
        np.save(savename+"/projections",projections)
        np.save(savename+"/correlations",correlations)