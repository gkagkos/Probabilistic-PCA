import numpy as np
from copy import copy
import pandas as pd
from sklearn.metrics import mean_squared_error



def nullcol(x): return (count(x)!=len(x)).index[count(x)!=len(x)].tolist()
def missingcol(x): return nullcol(x)

def count(x):
    try: return x.count()
    except: return len(x)


def table(x): 
    try: return pd.DataFrame(x)
    except: return pd.DataFrame(list(x.items()))

def bpca_complete(x, epochs = 100):
    decimals = 4
    y = copy(x)
    cols = y.columns.tolist()
    maximum = np.int(np.max(y.max())*999)
    means = round(y.mean(),decimals); sd = round(y.std(),decimals); y = round((y-means)/sd,decimals)
    y[missingcol(y)] = y[missingcol(y)].fillna(maximum)
    mat = np.matrix(y)

    N,d = mat.shape
    q = d-1
    yest = np.copy(mat); yest[yest==maximum]=0

    missidx = {};       bad = np.where(mat==maximum)
    for a in bad[0]: missidx[a] = []
    for a in range(len(bad[0])): missidx[bad[0][a]].append(bad[1][a])

    nomissidx = {};     good = np.where(mat!=maximum)
    for a in good[0]: nomissidx[a] = []
    for a in range(len(good[0])): nomissidx[good[0][a]].append(good[1][a])

    gmiss = list(set(bad[0]))
    gnomiss = list(set(good[0]))

    covy = np.cov(yest.T)
    U, S, V = np.linalg.svd(np.matrix(covy))
    U = (U.T[0:q]).T;         S = S[0:q]*np.eye(q);           V = (V.T[0:q]).T

    mu = np.copy(mat);        mu[mu==maximum]=np.nan;        mu = np.nanmean(mu, 0)
    W = U*np.sqrt(S);         tau = 1/ (np.trace(covy)-np.trace(S));      taumax = 1e20; taumin = 1e-20;    tau = np.amax([np.amin([tau,taumax]),taumin])

    galpha0 = 1e-10;          balpha0 = 1;                 alpha = (2*galpha0 + d)/(tau*np.diag(W.T*W)+2*galpha0/balpha0)
    gmu0  = 0.001;            btau0 = 1;                   gtau0 = 1e-10;                   SigW = np.eye(q)
    tauold = 1000

    for epoch in range(epochs):
        Rx = np.eye(q)+tau*W.T*W+SigW;            Rxinv = np.linalg.inv(Rx)
        idx = gnomiss; n = len(idx)                  
        dy = mat[idx,:] - np.tile(mu,(n,1));      x = tau * Rxinv * W.T * dy.T

        Td = dy.T*x.T;                            trS = np.sum(np.multiply(dy,dy))
        for n in range(len(gmiss)):
            i = gmiss[n]
            dyo = np.copy(mat)[i,nomissidx[i]] - mu[nomissidx[i]]
            Wm = W[missidx[i],:];                                  Wo = W[nomissidx[i],:]
            Rxinv = np.linalg.inv( Rx - tau*Wm.T*Wm );             ex = tau * Wo.T * np.matrix(dyo).T;   x = Rxinv * ex
            dym = Wm * x;                                          dy = np.copy(mat)[i,:]
            dy[nomissidx[i]] = dyo;                                dy[missidx[i]] = dym.T
            yest[i,:] = dy + mu
            Td = Td + np.matrix(dy).T*x.T;                            Td[missidx[i],:] = Td[missidx[i],:] + Wm * Rxinv
            trS = trS + dy*np.matrix(dy).T +  len(missidx[i])/tau + np.trace( Wm * Rxinv * Wm.T )

        Td = Td/N;                trS = trS/N;                        Rxinv = np.linalg.inv(Rx); 
        Dw = Rxinv + tau*Td.T*W*Rxinv + np.diag(alpha)/N;             
        Dwinv = np.linalg.inv(Dw)
        W = Td * Dwinv

        tau = (d+2*gtau0/N)/(trS-np.trace(Td.T*W)  + (mu*np.matrix(mu).T*gmu0+2*gtau0/btau0)/N)[0,0]
        SigW = Dwinv*(d/N)
        alpha = (2*galpha0 + d)/ (tau*np.diag(W.T*W)+np.diag(SigW)+2*galpha0/balpha0).T

        if np.abs(np.log10(tau)-np.log10(tauold)) < 1e-4:  
            break
        tauold = tau
    out = table(yest)
    out.columns = cols
    out = (out*sd)+means

    return out


if __name__ == '__main__':
    
    df = pd.read_csv("sample999.dat", header=None, sep = "\s+")
    df =df.replace(999.0, np.nan)
    new_df = bpca_complete(df,100)
    print(new_df)
    original_df = pd.read_csv("sampleorg.dat", header= None, sep ="\s+")

    rms = np.sqrt(mean_squared_error(original_df, new_df))
    print(rms)
