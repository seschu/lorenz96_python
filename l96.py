#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.integrate import odeint
import scipy.linalg.lapack as lapack
import numpy as np
import matplotlib.pylab as pylab
import scipy.linalg as linalg
from scipy import triu
import os
import matplotlib.pyplot as plt


def setupL96(N,F):    
    #
    # This section contains general stuff about the Lorenz 96 model and its derivatives
    #
    
    dimN = N
    
    I=np.arange(0,N,1)
    Im2=(I-2) % N
    Im1=(I-1) % N
    Ip1=(I+1) % N
    #Ip2=(I+2) % N
    
    L96 = lambda x,t: (x[Ip1]-x[Im2]) * x[Im1]- x[I] + F
    L96Jac = lambda x,t: np.stack((-x[Im1],x[Ip1]-x[Im2],-1.0*np.ones(N),x[Im1]),axis = 1)
    L96JacV = lambda x,v,t : np.multiply(v[Im2,:],-x[Im1,np.newaxis])+np.multiply(v[Im1,:],x[Ip1,np.newaxis]-x[Im2,np.newaxis])-v[I,:]+np.multiply(v[Ip1,:],x[Im1,np.newaxis])
    
    #L96Full = lambda t,x: np.concatenate((L96(x[0:dimN],t), np.reshape(L96JacV(x[0:dimN],np.reshape(x[dimN:],(dimN,dimN)),t),((dimN)**2,1)))
    
    #L96Hes = np.stack(((0.0,-1.0,0.0,0.0),(-1.0,0.0,0.0,1) ,(0.0,0.0,0.0,0.0),(0.0,1.0,0.0,0.0)), axis = 0)
    L96JacFull = None
    return L96,L96Jac,L96JacV,L96JacFull,dimN


def setupL96_2layer(para):    
    #
    # This section contains general stuff about the Lorenz 96 model and its derivatives
    #
    
    
    rescale = para['RescaledY']
    dimX = para['dimX'] 
    dimY = para['dimY']
    h    = para['h']
    c    = para['c']
    b    = para['b']
    F1   = para['F1']
    F2   = para['F2']
    
    NX = dimX
    NY = dimX*dimY    
    dimN = NX+NY
    
    XI=np.array(range(0,NX,1))
    XIm2=(XI-2) % NX
    XIm1=(XI-1) % NX
    XIp1=(XI+1) % NX
    #XIp2=(XI+2) % NX
    
    
    YI=np.array(range(0,NY,1))
    YIm2=(YI-2) % NY + NX
    YIm1=(YI-1) % NY + NX
    YIp1=(YI+1) % NY + NX
    YIp2=(YI+2) % NY + NX
    YI = YI + NX
    
    IndexHelp = list(map(int,list(np.floor((YI-NX)/dimY) )))
    if rescale: 
        facx = h*c/b/dimY
        facy = h*c/b
    else:
        facx = h*c/b
        facy = h*c/b
    
    XL96 = lambda x,t:     (x[XIp1]-x[XIm2]) * x[XIm1]- x[XI] + F1 - facx * np.sum(np.reshape(x[YI],(dimY,dimX), order = "F"),axis=0)
    YL96 = lambda x,t: -c*b*x[YIp1]*(x[YIp2]-x[YIm1])- c*x[YI] + F2 + facy * x[IndexHelp]
    L96 = lambda x,t: np.concatenate((XL96(x,t), YL96(x,t)))
    
    L96JacV = lambda x,v,t : np.concatenate(( np.multiply(v[XIp1,:],x[XIm1,np.newaxis])+np.multiply(v[XIm2,:],-x[XIm1,np.newaxis])+np.multiply(v[XIm1,:],x[XIp1,np.newaxis]-x[XIm2,np.newaxis])-  v[XI,:] - facx*np.sum(np.reshape(v[YI,:],(dimY,dimX,v.shape[1]), order = "F"),axis=0),
                                         -c*b*(np.multiply(v[YIp1,:],x[YIp2,np.newaxis]-x[YIm1,np.newaxis])+np.multiply(v[YIm2,:],x[YIp1,np.newaxis])+np.multiply(v[YIm1,:],-x[YIp1,np.newaxis]))-c*v[YI,:] + facy*v[IndexHelp,:]))
    #XL96JacV = lambda x,v,t: np.multiply(v[XIp1,:],x[XIm1,np.newaxis])+np.multiply(v[XIm2,:],-x[XIm1,np.newaxis])+np.multiply(v[XIm1,:],x[XIp1,np.newaxis]-x[XIm2,np.newaxis])-  v[XI,:] - fac/dimY*np.sum(np.reshape(v[YI,:],(dimY,dimX,v.shape[1]), order = "F"),axis=1)    
    #YL96JacV = lambda x,v,t: -c*b*(np.multiply(v[YIp1,:],x[YIp2,np.newaxis]-x[YIm1,np.newaxis])+np.multiply(v[YIm2,:],x[YIp1,np.newaxis])+np.multiply(v[YIm1,:],-x[YIp1,np.newaxis]))-c*v[YI,:] + fac*v[IndexHelp,:]
    
    
    #def L96(vec,t):
    #    tend = np.zeros(vec.shape[0])
    #    for i in range(0,dimX):
    #        tend[0:dimX] = (vec[XIp1[i]]-vec[XIm2[i]]) * vec[XIm1[i]]
    #        - vec[XI[i]] + F1 - fac * np.sum(vec[dimX+(i-1)*dimY:dimX+i*dimY])
    #    for i in range(0,NY):
    #        tend[i+NX] = -c*b*vec[YIp1[i]]*(vec[YIp2[i]]-vec[YIm1[i]])- c*vec[YI[i]] + F2 + fac*vec[IndexHelp[i]]
    #    return tend
    
#    def L96Hess:
#        for i in range(0,NX+NY)
        
        
#    def L96JacFull(vec,t):
#        dim=vec.shape[0]
#        jac = np.zeros((dim,dim))
#        for i in range(0,vec.shape[0]):
#            if i<dimX:
#                jac[i,XIp1[i]] = vec[XIm1[i]]
#                jac[i,XIm2[i]] = -vec[XIm1[i]]
#                jac[i,XIm1[i]] = vec[XIp1[i]]-vec[XIm2[i]]
#                jac[i,XI[i]] = -vec[XI[i]]
#                jac[i,dimX+(i-1)*dimY:dimX+i*dimY]=-facx
#            elif i>=dimX and i<NX+NY:
#                jac[i,YIp2[i]] = -c*b*vec[YIp1[i]]
#                jac[i,YIm1[i]] = +c*b*vec[YIp1[i]]
#                jac[i,YIp1[i]] = -c*b*vec[YIp2[i]]-vec[YIm1[i]]
#                jac[i,YI[i]] = -c*vec[YI[i]]
#                jac[i,IndexHelp[i]] = facy
#            else:
#                I=i-NX-NY % (NX+NY)
#                if I<dimX:
#                    jac[i,j]=np.multiply(HessMat[I,:,:],vec)
#                else:
#                
#        return jac
#    
#    def L96Jac(vec,t):
#        jac = np.zeros((vec.shape[0],vec.shape[0]))
#        for i in NX+NY:
#            if i<dimX:
#                jac[i,XIp1[i]] = vec[XIm1[i]]
#                jac[i,XIm2[i]] = -vec[XIm1[i]]
#                jac[i,XIm1[i]] = vec[XIp1[i]]-vec[XIm2[i]]
#                jac[i,XI[i]] = -vec[XI[i]]
#                jac[i,dimX+(i-1)*dimY:dimX+i*dimY]=-facx
#            elif i>=dimX and i<NX+NY:
#                jac[i,YIp2[i]] = -c*b*vec[YIp1[i]]
#                jac[i,YIm1[i]] = +c*b*vec[YIp1[i]]
#                jac[i,YIp1[i]] = -c*b*vec[YIp2[i]]-vec[YIm1[i]]
#                jac[i,YI[i]] = -c*vec[YI[i]]
#                jac[i,IndexHelp[i]] = facy
#        return jac
    
#    L96JacRef=L96Jac(np.random.rand(NX+NY),t)

#    HessMat=np.zeros(NX+NY,NX+NY,NX+NY)
#    for k in range(0,NX):
#        for i in range(0,NX):
#            for j in range(0,NX):
#                HessMat[XI[k],XIp1[k],XIm1[j])=1
#                HessMat[XI[k],XIm1[j],XIp1[k])=1
#                HessMat[XI[k],XIm2[k],XIm1[j])=-1
#                HessMat[XI[k],XIm1[j],XIm2[k])=-1
#    for k in range(0,NY):
#        for i in range(0,NY):
#            for j in range(0,NY)
#                HessMat[YI[k],YIp1[k],YIm1[j])=1
#                HessMat[YI[k],YIm1[j],YIp1[k])=1
#                HessMat[YI[k],YIm2[k],YIm1[j])=-1
#                HessMat[YI[k],YIm1[j],YIm2[k])=-1    
#    
    
        
    L96Jac=None
    L96JacFull=None
    return L96,L96Jac,L96JacV,L96JacFull,dimN


#
# Define a data type for Lorenz 96
#


class GinelliForward():
    import scipy.linalg.lapack as lapack
    import numpy as np

    def __init__(self, dimN, dimM, tendfunc=None, jacfunc=None, jacVfunc=None, jacfull=None, ml=None, mu=None):
        self.N = dimN
        self.M = dimM
        if dimM > dimN: quit("ERROR: M cannot be larger than N")
        dtype = np.dtype([('back', np.float64, dimN), ('lin', np.float64, (dimN,dimM)), ('tau', np.float64, dimM ), ('qr' , np.bool)])

        # stores last time step
        self.xold = np.array((np.zeros(dimN) , np.zeros((dimN,dimM)), np.zeros(dimM), False ), dtype=dtype)

        # current state of the linear and non-linear fields
        self.x = np.array((np.zeros(dimN) , np.zeros((dimN,dimM)), np.zeros(dimM), False ), dtype=dtype)

        
        # blocksize for dgeqrf (see LAPACK documentation)
        [QR,tau,work,info]=lapack.dgeqrf(np.random.rand(self.N,self.M),lwork=-1)
        self.NB_lapack = work[0] 
        [res,work,info]=lapack.dormqr('L','N',QR,tau,np.random.rand(self.N,self.M),lwork=-1)
        self.NB_dormqr = work[0] 

        # this counts the number of steps taken
        self.step=0

        # time t belonging to step 
        self.step_t = 0.0 


        # this is the tendency of the non-linear background
        self.tendency = tendfunc 

        # this is the tendency of the linear equation, hence the jacobian
        self.jacobian = jacfunc 

        # jacobian of linear, non-linear system
        self.jacobianfull = jacfull

        # lyapunov exponents
        self.lyap = np.zeros(dimM)

        # full tendency of simultaneously computing linear and non-linear tendency
        self.fulltend = lambda x,t: np.concatenate((tendfunc(x[0:dimN],t),jacVfunc(x[0:dimN],np.reshape(x[dimN:],(dimN,dimM), order = 'F'),t).flatten(order = 'F')))
        
        # this is for optimization purposes of odeint
        self.ml = ml
        
        # this is for optimization purposes of odeintself.step_t = self.step_t + delta_t 
       
        self.mu = mu 
    
        self.equilibrium = None
    
    def set_equilibrium(self,eq):
        # set equilibrium for this equation
        self.equilibrium = eq
    
    # initialize the background state
    def init_back(self,init = 'equilibrium',scale = 1.0):
        if isinstance(init,str):
            if init.lower() == 'equilibrium': self.x['back'] = self.equilibrium 
            elif init.lower() == 'random' : self.x['back'] = np.random.rand(self.N)*scale
            else: quit('ERROR in class L96, init_back')
        else: self.x['back'] = init
    
    # initialize the linear ensemble
    def init_lin(self,init = 'random',scale = 1.0):
        if isinstance(init,str):
            if init.lower() == 'random' :
                self.x['lin'] = np.random.rand(self.N,self.M)*scale
                self.qr()
            else: quit('ERROR in class L96, init_lin')
        else: self.x['lin'] = init
    
    # QR decompose the linear perturbations with unique positive dimensions
    def qr(self):
        [QR,self.x['tau'],work,info]=lapack.dgeqrf(self.x['lin'],lwork=self.N*self.NB_lapack)
        self.x['qr'] = True
        self.x['lin'] = QR

    # performs QR step necessary for Lyapunov computation
    def qr_step(self,delta_t):
        self.integrate_all(delta_t,normalize=False)
        self.qr()
        self.lyap = np.log(np.abs(self.diagR()))/delta_t
        self.PackedR=self.GetPackedR()
        self.R=self.GetUnPackedR()
        self.restoreQ() 
        
    # returns diagonal of R matrix
    def diagR(self):
        res = np.diagonal(self.x['lin'])
        return res 

    # multiplies the lineL96JacVar matrix with a matrix Mat from side S 
    def matmul_lin(self,Mat, S = 'L', TRANS_X = 'N', TRANS_Q = 'N'):
        if self.x['qr']: 
            if TRANS_X.upper() == 'T': inp = np.transpose(Mat)
            elif TRANS_X.upper() == 'N': inp = Mat
            res,_,info = lapack.dormqr(S.upper(), TRANS_X.upper(), self.x['lin'], self.x['tau'], inp, self.N*self.NB_dormqr)
        else:
            if S.upper() == 'L': res=lapack.dgemm(1.0, self.x['lin'], Mat, trans_a = TRANS_Q.upper(), trans_b = TRANS_X.upper())                       
            elif S.upper() == 'R': res=lapack.dgemm(1.0, Mat, self.x['lin'], trans_a = TRANS_X.upper(), trans_b = TRANS_Q.upper())
        return res

    # get R from QR matrix with 
    def GetPackedR(self):
        if not self.x['qr']: raise ValueError('Cannot make getpackedR since no QR has been applied.')
        R = np.zeros(np.int(self.M*(self.M+1)/2))       
        for j in range(0,self.M):
            for i in range(0,j):
                R[np.int(i + j*(j + 1)/2 -1)] = self.x['lin'][i, j]         
        return R
        
    # get unpacked R from QR matrix with 
    def GetUnPackedR(self):
        if not self.x['qr']: raise ValueError('Cannot make getpackedR since no QR has been applied.')
        R = np.zeros((self.N,self.M))        
        for j in range(0,self.M):
            for i in range(0,j+1):
                R[i,j] = self.x['lin'][i, j]         
        return R
        
    # recreate Q fully 
    def restoreQ(self):
        if not self.x['qr']: raise ValueError('Cannot make restoreQ since no QR has been applied.')
        res,_,info = lapack.dormqr('L', 'N', self.x['lin'], self.x['tau'], np.eye(self.N,self.M), self.N*self.NB_lapack,overwrite_c=0)
        self.x['qr'] = False
        self.x['lin'] = res[:,0:self.M]
        return res,info

    # tests whether the QR decomposition succesfully delivered an orthogonal matrix
    def testQ(self):
        res,_,_ = lapack.dormqr('L', 'N', self.x['lin'], self.x['tau'], np.eye(self.M,self.N), self.N*self.NB_lapack)
        return np.allclose(np.matmul(res,np.transpose(res)), np.eye(self.M,self.M))
        
    # integrate only background
    def integrate_back(self,delta_t):
        if not isinstance(delta_t,float): tarray=[self.step_t,self.step_t+delta_t]
        else: tarray=delta_t
        intres = odeint(self.tendency, self.x['back'], tarray)#, Dfun = self.jacobian)#, ml = self.ml,mu = self.mu)
        self.xold['back'] = intres[0,:self.N]
        self.x['back'] = intres[-1,:self.N]
        self.step_t = self.step_t + delta_t 
        return intres
    
    # integrate background and tangent linear at the same time
    def integrate_all(self, delta_t, normalize = False):
        if self.x['qr']: raise ValueError('Cannot integrate qr decomposed ensemble, use restoreQ first')
        intres = odeint(self.fulltend, np.concatenate((self.x['back'] , self.x['lin'].flatten(order = 'F'))), [self.step_t,self.step_t+delta_t], Dfun = self.jacobianfull) #, ml = self.ml,mu = self.mu)
        self.xold['back'] = intres[0,:self.N]
        self.xold['lin'] = np.reshape(intres[0,self.N:],(self.N,self.M),order = 'F')
        self.x['back'] = intres[1,:self.N]
        self.x['lin'] = np.reshape(intres[1,self.N:],(self.N,self.M),order = 'F')
        self.step_t = self.step_t + delta_t 
        self.step = self.step + 1
        if normalize: self.x['lin'] = self.x['lin']/np.linalg.norm(self.x['lin'],axis=0,keepdims=True)

   
