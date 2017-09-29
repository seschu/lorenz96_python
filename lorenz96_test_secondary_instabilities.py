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

precision = 'float64'

scalar = lambda x,y : np.sum(x*y, dtype = precision )
norm = lambda x : np.sqrt(scalar(x,x), dtype = precision )
normalize = lambda x : x/norm(x)
corr = lambda x,y: scalar(normalize(x),normalize(y))

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
           '2lay' : True,
           'integrator': 'classic'
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
           'time' : np.arange(0,500,0.05),
           'spinup' : 100,
           '2lay' : False,
           'integrator': 'classic'
           }


testzeroclv=True

hs=[1.0] #, 0.5] #   ,  0.0625,  0.125 ,  0.25  ,  0.5   ,  1.    ]
experiments = [paraL96_1lay]#,paraL96_2lay]#,paraL96_1lay]    
integrator = 'classic'

# first test clv
epsilons=10.0**np.arange(0,2.1,0.2,dtype = precision)
intsteps=np.arange(0,100,1)


CLVs=[1,2,3,4,5] # ,13, 30]#np.arange(1,2,1)
timeintervall = range(200,600,100)
for paraL96,h in product(experiments ,hs):
    if not paraL96['2lay'] and not h == 1.0: print("1 lay only with h = 1.");break
    savename=paraL96['expname']+"_h_"+str(h)
    spinup = paraL96['spinup']        
    paraL96=np.load(savename+"/paraL96.npy")
    paraL96=paraL96[()]
    dt = 0.005#np.mean(np.diff(paraL96['time']))
    # M number exponents
    if paraL96['2lay']:
        M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
        dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
    else:
        M = paraL96['dimX'] 
        dimN = paraL96['dimX'] 
    
    
    steplengthforsecondorder = np.load(savename+'/steplengthforsecondorder.npy')
    
    np.save(savename+"/CLVs",CLVs)
    dtau=np.mean(np.diff(paraL96['time']))
    t = paraL96['time'][timeintervall]
    correlation=[]
    correlationv2=[]
    correlationv3=[]
    realgrowth=[]
    normerror = []
    normerror_1st = []
    normerrorrel = []
    normerrorrel_1st = []
    normerrorrel_2nd = []
    normnonlin =[]
    
    for clv in CLVs:
        correlation.append(np.memmap(savename+'/correlation_only_1_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        correlationv2.append(np.memmap(savename+'/correlation_1and2_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        correlationv3.append(np.memmap(savename+'/correlation_only_2_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        normerror.append(np.memmap(savename+'/normerror_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        normerror_1st.append(np.memmap(savename+'/normerror_1st_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        normerrorrel_1st.append(np.memmap(savename+'/normerrorrel_1st_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        normerrorrel_2nd.append(np.memmap(savename+'/normerrorrel_2nd_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        normerrorrel.append(np.memmap(savename+'/normerrorrel_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        normnonlin.append(np.memmap(savename+'/normnonlin_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
        realgrowth.append(np.memmap(savename+'/realgrowth_clv'+str(clv)+'.dat',mode='w+',shape=(len(timeintervall),len(epsilons),len(intsteps)-1),dtype=precision,order = 'C'))
 
    CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(paraL96['time']),dimN,M),dtype='float64',order = 'C')
    lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(paraL96['time']),M),dtype='float64',order = 'C')
    trajectory = np.memmap(savename+'/trajectory.dat',mode='r',shape=(len(paraL96['time']),dimN),dtype='float64',order = 'C')
    full_solution = np.memmap(savename+'/full_solution.dat',mode='r',shape=(len(paraL96['time']),len(steplengthforsecondorder),dimN,M),dtype='float64',order = 'C')
    
    if paraL96['2lay']: L96,L96Jac,L96JacV,L96JacFull,dimN = l96.setupL96_2layer(paraL96)
    else: L96,L96Jac,L96JacV,L96JacFull,dimN = l96.setupL96(paraL96)
    
    field = l96.GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=paraL96['integrator'])
    field.rtol = 1e-8
    field.atol = 1e-8
    for i,tn in enumerate(timeintervall):
        print(tn)
        
        for en,epsilon in enumerate(epsilons):
            print(' eps: '+str(epsilon))
            for nc, clv in enumerate(CLVs):
                field.x['back']=trajectory[tn,:]+epsilon*CLV[tn,:,clv-1]
                #print(0,np.log10(epsilon),np.log10(np.linalg.norm(epsilon*CLV[tn,:,clv-1])))
                    
                for step, (a,b) in enumerate(zip(intsteps[0:-1],intsteps[1:])):
                    stepsize=b
                    #print('g')
                    for counting in np.arange(0,b-a):
                        field.integrate_back(dtau,dt =dt)
                        #print('counting')
                    nonlinear_prediction = field.x['back'] - trajectory[tn+stepsize,:]
                    firstorder_prediction    = CLV[tn+stepsize,:,clv-1]*np.exp(dtau*np.memmap.sum(lyaploc_clv[tn:tn+stepsize,clv-1]))*epsilon
                    secondorder_prediction    = full_solution[tn,stepsize,:,clv-1]*epsilon**2.0
                    secondorder_prediction_wo2e    = full_solution[tn,stepsize,:,clv-1]
                    
                    firstandsecondorder_prediction = (full_solution[tn,stepsize,:,clv-1]*epsilon**2.0+
                                                      epsilon*CLV[tn+stepsize,:,clv-1]*
                                                      np.exp(dtau*np.memmap.sum(lyaploc_clv[tn:tn+stepsize,clv-1])))
                    firstandsecondorder_prediction_woe = (full_solution[tn,stepsize,:,clv-1]*epsilon
                                                      +CLV[tn+stepsize,:,clv-1]
                                                      *np.exp(dtau*np.memmap.sum(lyaploc_clv[tn:tn+stepsize,clv-1])))
                    
                    #print(step+1,np.log10(epsilon),np.log10(np.linalg.norm(nonlinear_prediction - firstorder_prediction)),np.log10(np.linalg.norm(nonlinear_prediction - firstandsecondorder_prediction)))
                    
                    correlation[nc][i,en,step] = corr(nonlinear_prediction,firstorder_prediction)
                    
                    correlationv2[nc][i,en,step] = corr(nonlinear_prediction,firstandsecondorder_prediction_woe)
                                                    #(scalar(nonlinear_prediction,firstorder_prediction) + 
                                                    #epsilon*scalar(nonlinear_prediction,secondorder_prediction_wo2e))/(
                                                    #        norm(nonlinear_prediction)*norm(firstandsecondorder_prediction_woe))
                    
                    correlationv3[nc][i,en,step] = corr(nonlinear_prediction,secondorder_prediction_wo2e)
                    
                    realgrowth[nc][i,en,step] = np.log(norm(nonlinear_prediction)/epsilon)/(dtau*(stepsize))
                    
                    normerror[nc][i,en,step] = norm(nonlinear_prediction - firstandsecondorder_prediction)
                    
                    normerror_1st[nc][i,en,step] = norm(nonlinear_prediction - firstorder_prediction)
                    
                    normerrorrel[nc][i,en,step] = norm(nonlinear_prediction - firstandsecondorder_prediction)/norm(nonlinear_prediction)
                    
                    normerrorrel_1st[nc][i,en,step] = norm(nonlinear_prediction - firstorder_prediction)/norm(nonlinear_prediction)
                    
                    normerrorrel_2nd[nc][i,en,step] = norm(nonlinear_prediction - secondorder_prediction)/norm(nonlinear_prediction)
                    
                    normnonlin[nc][i,en,step] = norm(nonlinear_prediction)
                    
                    
                    
        if i % 50 == 0:
            for nc, clv in enumerate(CLVs):
                np.memmap.flush(realgrowth[nc])
                np.memmap.flush(correlation[nc])
                np.memmap.flush(correlationv2[nc])
                np.memmap.flush(correlationv3[nc])
                np.memmap.flush(normerror[nc])
                np.memmap.flush(normerror_1st[nc])
            print("flushed")

    for nc, clv in enumerate(CLVs):    
        np.memmap.flush(realgrowth[nc])
        np.memmap.flush(correlation[nc])
        np.memmap.flush(correlationv2[nc])
        np.memmap.flush(correlationv3[nc])
        np.memmap.flush(normerror[nc])
        np.memmap.flush(normerror_1st[nc])

    np.save(savename+"/timeintervall", timeintervall)
    np.save(savename+"/epsilons",epsilons)        
    np.save(savename+"/intsteps",intsteps)        
    print("Saveing results in folder "+savename+".")
