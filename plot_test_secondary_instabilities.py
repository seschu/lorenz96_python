#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg
from itertools import product
#from sklearn import preprocessing
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)

normalize = lambda x : x/np.sqrt(np.sum(x**2.0))

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
if not os.path.exists(resultsfolder): os.mkdir(resultsfolder)
    
hs=[1.0]#, 0.5] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
experiments = [paraL96_1lay]    
integrator = 'classic'

# first test clv
min_epsilon=10**-9
max_epsilon=1


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
        dtau = np.diff(paraL96['time']).mean()
        
        growth = np.memmap(savename+'/growth.dat',mode='r',shape=(len(t),lensteparray,M),dtype='float64')
        lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M),dtype='float64')
        
        if not os.path.exists(savename): os.mkdir(savename)
    
        for fold in ['erroranalysis']:
            if not os.path.exists(resultsfolder+"/"+fold): os.mkdir(resultsfolder+"/"+fold)
        
        precisioncorr = 'float64'        
        for clv in CLVs:
            correlation = np.memmap(savename+'/correlation_only_1_clv'+str(clv)+'.dat',mode='r',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precisioncorr,order = 'F')
            correlationv2 = np.memmap(savename+'/correlation_1and2_clv'+str(clv)+'.dat',mode='r',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precisioncorr,order = 'F')
            correlationv3 = np.memmap(savename+'/correlation_only_2_clv'+str(clv)+'.dat',mode='r',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precisioncorr,order = 'F')
            realgrowth = np.memmap(savename+'/realgrowth_clv'+str(clv)+'.dat',mode='r',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precisioncorr,order = 'F')
            normerror = np.memmap(savename+'/normerror_clv'+str(clv)+'.dat',mode='r',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precisioncorr,order = 'F')
            normnonlin = np.memmap(savename+'/normnonlin_clv'+str(clv)+'.dat',mode='r',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precisioncorr,order = 'F')
            
            CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(paraL96['time']),dimN,M),dtype='float64')
            lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(paraL96['time']),M),dtype='float64')
            trajectory = np.memmap(savename+'/trajectory.dat',mode='r',shape=(len(paraL96['time']),dimN),dtype='float64')
        
            
            imin = np.where(epsilons==min_epsilon)[0][0]
            imax = np.where(epsilons==max_epsilon)[0][0]+1
            X,Y = np.meshgrid(np.log10(epsilons[imin:imax]),intsteps[1:]*dtau)
            
        
            maskcorr = np.load(savename+"/maskcorr.npy")
            fig=plt.figure()
            plt.contourf(X,Y,np.transpose(np.mean(np.abs(correlation)[:,imin:imax,:],axis =0)),np.arange(0, 1.1, .1))
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.colorbar()
            plt.title('Average Correlation of non-linear perturbation\n after time t (y axis) along CLV '+str(clv)+" using 1st order")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_correlation_first_order.pdf")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_correlation_first_order.png", dpi =400)
            plt.close(fig)

            maskcorr = np.load(savename+"/maskcorr.npy")

            fig=plt.figure()
            plt.contourf(X,Y,np.transpose(np.mean(np.abs(correlationv2)[:,imin:imax,:],axis =0)),np.arange(0, 1.1, .1))
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.colorbar()
            plt.title('Average Correlation of non-linear perturbation\n after time t (y axis) along CLV '+str(clv)+" using 1st and 2nd order")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_correlation_firstandsecond_order.pdf")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_correlation_firstandsecond_order.png", dpi =400)
            plt.close(fig)
            
            fig=plt.figure()
            plt.contourf(X,Y,np.transpose(np.mean(np.abs(correlationv3)[:,imin:imax,:],axis =0)),np.arange(0, 1.1, .1))
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.colorbar()
            plt.title('Average Correlation of non-linear perturbation\n after time t (y axis) along CLV '+str(clv)+" using only 2nd order")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_correlation_second_order.pdf")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_correlation_second_order.png", dpi =400)
            plt.close(fig)
            
            fig=plt.figure()
            le = np.mean(lyaploc_clv[timeintervall,clv-1], axis =0)
            levels = [le,100.*np.abs(le)] # [le-np.abs(le)*0.1,le,le+np.abs(le)*0.1]
            im1p = plt.contour(X,Y,np.transpose(np.mean(realgrowth[:,imin:imax,:],axis =0)),levels=levels,colors=('k'),linestyles=('-'),linewidths=(2,))
            im1 = plt.contourf(X,Y,np.transpose(np.mean(realgrowth[:,imin:imax,:],axis =0)))
            plt.clabel(im1p, fmt = '%2.2f', colors = 'k', fontsize = 14)
            fig.colorbar(im1)
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.title('Growth rate of non-linear perturbation\n after time t (y axis) along CLV '+str(clv)+" \n "+r"($\lambda$ = "+"{0:2.2f}".format(le[()])+") using 1st and 2nd order")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_growthnonlinear.pdf")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_growthnonlinear.png", dpi =400)
            plt.close(fig)
        
            fig=plt.figure()
            im2 = plt.contourf(X,Y,np.log(np.transpose(np.mean(normerror[:,imin:imax,:], axis = 0))))
            fig.colorbar(im2)
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.title('Log of ratio of norm error of 1st and 2nd order \n and norm of nonlinear prediction \n after time t (y axis) along CLV '+str(clv))
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_normerror.pdf")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_normerror.png", dpi =400)
            #fig.tight_layout()
            plt.close(fig)
            
            fig=plt.figure()
            for ist in range(0,normerror.shape[2]):
                plt.plot(np.log(epsilons),np.log(np.mean(normerror[:,:,ist], axis = 0))/intsteps[ist],label=str(intsteps[ist]))
            plt.xlabel(r'$\log(\epsilon)')
            plt.ylabel(r'size of error')
            plt.legend()
            plt.title('Log of relative difference between 1st and 2nd order \n and norm of nonlinear prediction \n after time t (y axis) along CLV '+str(clv))
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_errorgrowth.pdf")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_errorgrowth.png", dpi =400)
            plt.close(fig)
            
            
            fig=plt.figure()
            im2 = plt.contourf(X,Y,np.log(np.transpose(np.mean(normnonlin[:,imin:imax,:], axis = 0))))
            fig.colorbar(im2)
            plt.xlabel(r'$log(\epsilon)$')
            plt.ylabel(r'time [MTU]')
            plt.title('Log of norm of nonlinear prediction \n after time t (y axis) along CLV '+str(clv))
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_normnonlin.pdf")
            fig.savefig(resultsfolder+"/erroranalysis/CLV_"+str(clv)+"_normnonlin.png", dpi =400)
            plt.close(fig)
        
#        contracted_CLVs = np.memmap(savename+'/contracted_clvs.dat',mode='r',shape=(len(paraL96['time']),dimN,M),dtype=precision) #(final_clv,init_clv)
#        fig=plt.figure()
#        plt.contourf(np.mean(contracted_CLVs[2000:8000,:,0:30],axis=0),axis =0)
#        plt.colorbar()
#        plt.plot(np.cumsum(contracted_CLVs[2000:8000,10,5])/np.arange(1,8000))