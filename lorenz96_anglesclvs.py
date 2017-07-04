
minloc = np.argmin(np.abs(lyapmean_clv))
angle_un_s = np.zeros(t.shape[0])
corr_un_s = np.zeros(t.shape[0])
angle_1 = np.zeros((t.shape[0],M-1))
corr_1 = np.zeros((t.shape[0],M-1))
for tn,ti in enumerate(t):
    q1, _ = np.linalg.qr(CLV[tn,:,0:minloc,0], mode='reduced')
    q2, _ = np.linalg.qr(CLV[tn,:,minloc+1:,0], mode='reduced')
    s=np.linalg.svd(np.matmul(np.transpose(q1),q2),full_matrices = 0, compute_uv=0)
    angle_un_s[tn]=np.arccos(s[0])
    corr_un_s[tn]=s[0]
    for c in range(0,M-1):
        q1, _ = np.linalg.qr(CLV[tn,:,0:c+1,0], mode='reduced')
        q2, _ = np.linalg.qr(CLV[tn,:,c+1:,0], mode='reduced')
        s=np.linalg.svd(np.matmul(np.transpose(q1),q2),full_matrices = 0, compute_uv=0)
        angle_1[tn,c]=np.arccos(s[0])
        corr_1[tn,c]=s[0]
        