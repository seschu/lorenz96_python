#!/home/user/anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt

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
M = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum
dimN = paraL96['dimX'] + paraL96['dimX']*paraL96['dimY'] # -1 full spectrum


t = np.arange(0,200,0.005)

hs=[ 1. ]

plt.style.use('ggplot')

savename='gabrielessetup_convergence'

CLV = np.memmap(savename+'/CLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV = np.memmap(savename+'/BLV.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R = np.memmap(savename+'/R.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv = np.memmap(savename+'/lyapmean_blv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyapmean_clv = np.memmap(savename+'/lyapmean_clv.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyaploc_clv = np.memmap(savename+'/lyaploc_clv',mode='r',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv = np.memmap(savename+'/lyaploc_blv',mode='r',shape=(len(t)-1,M,len(hs)),dtype='float64')

CLV2 = np.memmap(savename+'/CLV2.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
BLV2 = np.memmap(savename+'/BLV2.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
R2 = np.memmap(savename+'/R2.dat',mode='r',shape=(len(t),dimN,M,len(hs)),dtype='float64')
lyapmean_blv2 = np.memmap(savename+'/lyapmean_blv2.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyapmean_clv2 = np.memmap(savename+'/lyapmean_clv2.dat',mode='r',shape=(M,len(hs)),dtype='float64')
lyaploc_clv2 = np.memmap(savename+'/lyaploc_clv2.dat',mode='r',shape=(len(t),M,len(hs)),dtype='float64')
lyaploc_blv2 = np.memmap(savename+'/lyaploc_blv2.dat',mode='r',shape=(len(t)-1,M,len(hs)),dtype='float64')

correlationclv = np.memmap(savename+'/correlationclv.dat',mode='r',shape=(len(t),M,len(hs)),dtype='float64')



from matplotlib import cm
figurename="convergence_clvs"
fig= plt.figure()
X, Y = np.meshgrid(range(50,300+1),t[0:-1])
contourf = plt.contourf(X, Y, correlationclv[0:-1,50:300+1,0])
ax=plt.gca()
fig.colorbar(contourf, ax=ax, shrink=0.9)
plt.show()
fig.savefig(savename+"/"+figurename+".png")
fig.savefig(savename+"/"+figurename+".pdf")
    

projX = np.memmap(savename+'/projX.dat',mode='r',shape=(len(t),M,len(hs)),dtype='float64')
projX2 = np.memmap(savename+'/projX2.dat',mode='r',shape=(len(t),M,len(hs)),dtype='float64')


for clv in range(0,dimN):
    print(clv)
    figurename="convergence_clv_"+str(clv+1)
    fig= plt.figure()
    plt.plot(t[0:-1],correlationclv[0:-1,clv,0],label='Correlation for CLV '+str(clv))
    R=np.sum(CLV[0:-1,0:paraL96['dimX'],clv,0]**2,axis=1)
    plt.plot(t[0:-1],R, label= 'Projection on X modes - first run')
    R=np.sum(CLV2[0:-1,0:paraL96['dimX'],clv,0]**2,axis=1)
    plt.plot(t[0:-1],R, label= 'Projection on X modes - second run')
    ax.set_xlabel('Time in model timeunits')
    ax.set_ylabel('Correlation')
    ax.set_title('Compareing two randomly initialized\n CLV Ginelli Computations')
    ax.set_ylim([0,1.01])
    plt.tight_layout()
    fig.savefig(savename+"/"+figurename+".png")
    fig.savefig(savename+"/"+figurename+".pdf")
    plt.close(fig)

fig, ax = plt.subplots(2, 1,  figsize=(8,8)) 
ax[0].plot(range(1,M+1),lyapmean_clv)
ax[0].plot(range(1,M+1),lyapmean_clv2)
ax[0].set_xlabel('Lyapunov Index')
ax[0].set_ylabel('Lyapunov Exponent [1/MTU]')
ax[0].set_title('Lyapunov Spectrum for Gabrieles Setup')
ax[1].plot(range(1,M+1),np.mean(projX[10000:30000,:],axis=0))
ax[1].set_xlabel('Lyapunov Index')
ax[1].set_ylabel('Lyapunov Exponent [1/MTU]')
ax[1].set_title('Projection onto Xmodes')
plt.tight_layout()
fig.savefig(savename+"/lyapspec.pdf")
fig.savefig(savename+"/lyapspec.png")