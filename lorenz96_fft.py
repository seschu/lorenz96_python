ax1.axvline(x=minloc, linewidth = 1, color = 'grey')
import matplotlib.pyplot as plt
import os
import numpy as np

testzeroclv=True



savename='secondaryinstabilities'


paraL96=np.load(savename+'/paraL96.npy')
paraL96=paraL96[()]
# M number exponents
M = paraL96['dimX'] #+ paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX']# + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimX = paraL96['dimX']# + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimY = paraL96['dimY']# + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum


CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV = np.memmap(savename+'/BLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R = np.memmap(savename+'/R.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='r',shape=(len(t)-1,M,len(hs)),dtype='float64')
if testzeroclv: tendency = np.memmap(savename+'/tendency.dat',mode='r',shape=(len(t),dimN,len(hs)),dtype='float64')
if testzeroclv: tendcorr = np.memmap(savename+'/tendcorr.dat',mode='r',shape=(len(t),len(hs)),dtype='float64')

corrs=np.arange(0.01,1,0.1)
lowest=np.zeros((corrs.shape[0],M))
length=np.zeros((corrs.shape[0],M))
for n,c in enumerate(corrs):
    d = (np.logical_and(np.abs(tendcorr[:,0])>c,np.abs(tendcorr[:,0])<c+0.1))
    length[n,:] = np.sum(d) 
    lowest[n,:] = np.average(lyaploc_clv[:,:,0],weights = d, axis = 0)

minloc = np.argmin(np.abs(lyapmean_clv))+1
maskcorr = (np.logical_and(np.abs(tendcorr[:,0])>0.95,np.abs(tendcorr[:,0])<1.01))

clvPSD=np.average(np.abs(np.fft.fft(CLV[:,:,:,0],axis=1)),weights = maskcorr, axis=0)

freq = np.arange(0,dimX/2) #np.fft.fftfreq(M,d=1)

X, Y = np.meshgrid(range(1,M+1),freq[:int(dimX/2)])

#f, (ax1, ax2) = plt.subplots(2,1, sharex = True)
#ax1.contourf(X,Y,clvPSD[:int(dimX/2),:])
#ax1.set_xlim(1,36)
#ax1.set_title("PSD")
#ax1.colorbar()
#ax2.axhline(y=0,xmin=0,xmax=36, linewidth = 1, color = 'grey')
#ax2.plot(np.arange(1,dimN+1),np.average(lyaploc_clv[:,:,0],weights = (maskcorr), axis = 0))
#ax2.set_xlim(1,36)
#ax2.set_title("Lyapunov Spectrum")

f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex = True, figsize=(8,20))
rescaledPSD = clvPSD[:int(dimN/2),:]/np.linalg.norm(clvPSD[:int(dimN/2),:],axis=0,keepdims=True)
im = ax1.contourf(X,Y,rescaledPSD)
#f.colorbar(im, ax=ax1)
ax1.set_xlim(1,36)
ax1.set_title("Rescaled PSD")
ax1.axvline(x=minloc, linewidth = 1, color = 'grey')
ax2.plot(np.arange(1,dimN+1),np.argmax(rescaledPSD, axis = 0))
ax2.set_title("Maximum Wavenumber")
ax2.set_ylim(5,20)
ax2.axvline(x=minloc, linewidth = 1, color = 'grey')
ax3.plot(np.arange(1,dimN+1),np.mean(np.multiply(freq[:,np.newaxis],rescaledPSD), axis = 0))
ax3.set_title("Mean Wavenumber")
ax3.axvline(x=minloc, linewidth = 1, color = 'grey')
ax4.axhline(y=0,xmin=0,xmax=36, linewidth = 1, color = 'grey')
ax4.axvline(x=minloc, linewidth = 1, color = 'grey')
ax4.plot(np.arange(1,dimN+1),np.average(lyaploc_clv[:,:,0],weights = (maskcorr), axis = 0))
ax4.set_xlim(1,36)
ax4.set_title("Lyapunov Spectrum")
f.tight_layout()


