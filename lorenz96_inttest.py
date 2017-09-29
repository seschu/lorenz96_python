#!/home/user/anaconda3/bin/python
import numpy as np
import os
import l96
from scipy import triu
import scipy.linalg as linalg

# these are our constants
paraL96 = {'F1' : 10,
           'F2' : 6,
           'b'  : 10,
           'c'  : 10,
           'h'  : 1,
           'dimX': 36,
           'dimY' : 10,
           'RescaledY' : False
           }

# M number exponents
M = 2 #paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
integrator = 'dopri5'
t = np.arange(0,10,0.05)
spinup = 100;
#setup L96
for key in paraL96.keys(): print(key+' : '+str(paraL96[key]))
L96,L96Jac,L96JacV,L96JacFull,dimN = l96.setupL96_2layer(paraL96)
field = l96.GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=integrator)
field2 = l96.GinelliForward(dimN,M,tendfunc = L96, jacfunc = L96Jac, jacVfunc = L96JacV,jacfull=L96JacFull, integrator=integrator)
# initialize fields 
print("\nInitialize ...")
field.init_back('random',0.1)
field2.init_back('random',0.1)
field.init_lin('random',0.1)
field.restoreQ()   
# spinup
print("\nSpinup ...")
#for i in range(0,int(spinup/0.1),1): 
# spinup
print("\ninitialize with same time step ...")
field.integrate_back(spinup)
field.step_t = 0.0
# save initial state
initial=field.x['back']
field2.step_t = 0.0

for integrator in ['classic', 'rk4','dopri','dop853']:
    field2.init_back(initial)
    field.init_back(initial)
    print("\n set integrator to "+integrator)
    field.defaultintegrator=integrator
    field2.defaultintegrator=integrator
    print("### only background ###")
    field.integrate_back(0.1,dt=0.001)
    field2.integrate_back(0.1,dt=0.001)
    print("\n Selbes Resultat in back?",(field2.x['back']==field.x['back']).all())
    print("### background and lin ###")
    field.integrate_all(0.1,dt=0.001)
    field2.integrate_all(0.1,dt=0.001)
    print("\n Selbes Resultat in back?",(field2.x['back']==field.x['back']).all())
#    print("\n Difference in back now",(field2.x['back']-field.x['back']))
#    print("\n Difference in back old",(field2.xold['back']-field.xold['back']))
#    print("\n Selbes Resultat in lin?",(field2.x['lin']==field.x['lin']).all())
#    print("\n Difference in lin now",(field2.x['lin']-field.x['lin']))
#    print("\n Difference in lin old",(field2.xold['lin']-field.xold['lin']))

print(" Different initialisation:")
for integrator in ['classic', 'rk4','dopri','dop853']:
    field2.init_back(field.x['back'])
    
    print("\n set integrator to "+integrator)
    field.defaultintegrator=integrator
    field2.defaultintegrator=integrator
    field.integrate_back(0.1,dt=0.001)
    field2.integrate_back(0.1,dt=0.001)
    print("\n Selbes Resultat in back?",(field2.x['back']==field.x['back']).all())
#    print("\n Selbes Resultat in back old?",(field2.xold['back']==field.xold['back']).all())
    print("### background and lin ###")
    field.integrate_all(0.1,dt=0.001)
    field2.integrate_all(0.1,dt=0.001)
    print("\n Selbes Resultat in back?",(field2.x['back']==field.x['back']).all())
 #   print("\n Difference in back now",(field2.x['back']-field.x['back']))
 #   print("\n Difference in back old",(field2.xold['back']-field.xold['back']))
 #   print("\n Selbes Resultat in lin?",(field2.x['lin']==field.x['lin']).all())
 #   print("\n Difference in lin now",(field2.x['lin']-field.x['lin']))
 #   print("\n Difference in lin old",(field2.xold['lin']-field.xold['lin']))
