#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product
from sklearn import preprocessing
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)

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


testzeroclv=True
resultsfolder = "secondaryinstabilities"

hs=[1.0, 0.5] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
experiments = [paraL96_1lay]    
integrator = 'classic'

# first test clv



for paraL96,h in product(experiments ,hs):
        if not paraL96['2lay'] and not h == 1.0: print("1 lay only with h = 1.");continue
        savename=paraL96['expname']+"_h_"+str(h)
        spinup = paraL96['spinup']        
        paraL96=np.load(savename+"/paraL96.npy")
        paraL96=paraL96[()]
        # M number exponents
        if paraL96['2lay']:
            M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
            dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
        else:
            M = paraL96['dimX'] 
            dimN = paraL96['dimX'] 
        
        steplengthforsecondorder = np.load(savename+'/steplengthforsecondorder.npy')    
        lensteparray = len(steplengthforsecondorder)    
        epsilons = np.load(savename+"/epsilons.npy")
        intsteps = np.load(savename+"/intsteps.npy")
        timeintervall = np.load(savename+"/timeintervall.npy")
        CLVs = np.load(savename+"/CLVs.npy")
        t = paraL96['time']
        dtau = np.diff(t).mean()
        
        measured=np.memmap(savename+'/measuredgrowth.dat',mode='r',shape=(len(CLVs),len(intsteps),len(paraL96['time'])),dtype='float64')
        growth = np.memmap(savename+'/growth.dat',mode='r',shape=(lensteparray,len(t),M),dtype='float64')
    
        for clv in CLVs:
            correlation = np.memmap(savename+'/correlation_clv'+str(clv)+'.dat',mode='r',shape=(len(intsteps),len(epsilons),len(t)),dtype='float64')
            correlationv2 = np.memmap(savename+'/correlationv2_clv'+str(clv)+'.dat',mode='r',shape=(len(intsteps),len(epsilons),len(t)),dtype='float64')
            realgrowth = np.memmap(savename+'/realgrowth_clv'+str(clv)+'.dat',mode='r',shape=(len(intsteps),len(epsilons),len(t)),dtype='float64')
        
            CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M),dtype='float64')
            lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M),dtype='float64')
            trajectory = np.memmap(savename+'/trajectory.dat',mode='r',shape=(len(t),dimN),dtype='float64')
        
            maskcorr = np.load(savename+"/maskcorr.npy")
            fig=plt.figure()
            X,Y = np.meshgrid(np.log10(epsilons),intsteps*dtau)
            plt.contourf(X,Y,np.mean(np.abs(correlation[:,:,timeintervall]),axis =2),np.arange(0, 1.1, .1))
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.colorbar()
            plt.title('Average Correlation of non-linear perturbation\n after time t (y axis) along CLV '+str(clv)+" using 1st order")
            fig.savefig(resultsfolder+"/CLV_"+str(clv)+"_correlation_first_order.pdf")
            fig.savefig(resultsfolder+"/CLV_"+str(clv)+"_correlation_first_order.png")
            plt.close(fig)
            maskcorr = np.load(savename+"/maskcorr.npy")
            fig=plt.figure()
            X,Y = np.meshgrid(np.log10(epsilons),intsteps*dtau)
            plt.contourf(X,Y,np.mean(np.abs(correlationv2[:,:,timeintervall]),axis =2),np.arange(0, 1.1, .1))
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.colorbar()
            plt.title('Average Correlation of non-linear perturbation\n after time t (y axis) along CLV '+str(clv)+" using 1st and 2nd order")
            fig.savefig(resultsfolder+"/CLV_"+str(clv)+"_correlation_firstandsecond_order.pdf")
            fig.savefig(resultsfolder+"/CLV_"+str(clv)+"_correlation_firstandsecond_order.png")
            plt.close(fig)
            fig, axarr = plt.subplots(2, figsize = (7,6))
            X,Y = np.meshgrid(np.log10(epsilons),intsteps*dtau)
            im1 = axarr[0].contourf(X,Y,np.mean(realgrowth[:,:,timeintervall],axis =2))
            fig.colorbar(im1, ax=axarr[0])
            axarr[0].set_xlabel(r'$log(\epsilon)$')
            axarr[0].set_ylabel(r'time [MTU]')
            axarr[0].set_title('Average growth of non-linear perturbation\n after time t (y axis) along CLV '+str(clv)+" using 1st and 2nd order")
            axarr[1].plot(intsteps*dtau,np.mean(growth[:,timeintervall,clv-1][intsteps,:], axis = 1))
            axarr[1].set_xlabel(r'time [MTU]')
            axarr[1].set_ylabel(r'growthrate in 1/MTU')
            axarr[1].set_title('Average growth of 1st and 2nd order sumemd \n after time t (y axis) along CLV '+str(clv)+" using 1st and 2nd order")
            fig.savefig(resultsfolder+"/CLV_"+str(clv)+"_growthnonlinear.pdf")
            fig.savefig(resultsfolder+"/CLV_"+str(clv)+"_growthnonlinear.png")
            fig.tight_layout()
            plt.close(fig)
        
#        contracted_CLVs = np.memmap(savename+'/contracted_clvs.dat',mode='r',shape=(len(t),dimN,M),dtype=precision) #(final_clv,init_clv)
#        fig=plt.figure()
#        plt.contourf(np.mean(contracted_CLVs[2000:8000,:,0:30],axis=0),axis =0)
#        plt.colorbar()
#        plt.plot(np.cumsum(contracted_CLVs[2000:8000,10,5])/np.arange(1,8000))