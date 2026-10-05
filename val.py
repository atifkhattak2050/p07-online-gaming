import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, make_classification, make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
rng=np.random.default_rng(1); V={}
# ---------- A: recovery of rule weights and noise rate
def best_thr(sc,yy,nq=300):
    o=np.argsort(sc); ss=sc[o]; yo=yy[o]; m=len(yo)
    c0=np.cumsum(yo==0); c1=np.cumsum(yo==1); c2=np.cumsum(yo==2)
    idx=np.unique(np.linspace(0,m-1,nq).astype(int)); best=(0,0,0)
    for i in idx:
        j=idx[idx>i]
        if len(j)==0: continue
        acc=c0[i]+c1[j]-c1[i]+c2[-1]-c2[j]; k=acc.argmax()
        if acc[k]>best[0]: best=(acc[k],ss[i],ss[j[k]])
    return best[0]/m,best[1],best[2]
import os
N=40034; rowsA=[]
for eps in ([] if os.path.exists('out/valA.csv') else [0,0.05,0.10,0.20,0.30]):
    for rep in range(10):
        S=rng.integers(0,20,N); D=rng.integers(10,180,N); L=rng.integers(1,100,N); A=rng.integers(0,50,N)
        wl=rng.uniform(0.5,3); wa=rng.uniform(1,5); sc=S*D+wl*L+wa*A
        t1,t2=np.quantile(sc,[rng.uniform(.2,.3),rng.uniform(.7,.8)])
        clean=np.where(sc>t2,2,np.where(sc>t1,1,0)); y=clean.copy(); fl=rng.random(N)<eps; y[fl]=rng.integers(0,3,fl.sum())
        bb=(0,)
        for gl in np.arange(0,3.01,0.5):
            for ga in np.arange(0,5.01,1.0):
                r=best_thr((S*D+gl*L+ga*A).astype(float),y)
                if r[0]>bb[0]: bb=(r[0],gl,ga,r[1],r[2])
        # refine around best
        gl0,ga0=bb[1],bb[2]
        for gl in np.arange(max(0,gl0-0.5),gl0+0.51,0.25):
            for ga in np.arange(max(0,ga0-1),ga0+1.01,0.25):
                r=best_thr((S*D+gl*L+ga*A).astype(float),y)
                if r[0]>bb[0]: bb=(r[0],gl,ga,r[1],r[2])
        s2=S*D+bb[1]*L+bb[2]*A; p=np.where(s2>bb[4],2,np.where(s2>bb[3],1,0))
        far=(np.abs(s2-bb[3])>150)&(np.abs(s2-bb[4])>250); agree=(p==y)[far].mean(); eh=1.5*(1-agree)
        rowsA.append(dict(eps=eps,rep=rep,wl=wl,wa=wa,wl_hat=bb[1],wa_hat=bb[2],eps_hat=eh,agree_clean=(p==clean).mean(),agree_obs=bb[0],ceiling=1-2*eps/3))
    print('eps',eps,flush=True)
A_=pd.read_csv('out/valA.csv') if not rowsA else pd.DataFrame(rowsA); A_.to_csv('out/valA.csv',index=False)
V['A']=A_.groupby('eps').agg(eps_hat_mean=('eps_hat','mean'),eps_hat_sd=('eps_hat','std'),agree_clean=('agree_clean','mean'),agree_clean_min=('agree_clean','min'),
     wl_err=('wl',lambda s: 0),).reset_index().to_dict('records')
for r in V['A']:
    sub=A_[A_.eps==r['eps']]; r['wl_abs_err']=float((sub.wl-sub.wl_hat).abs().mean()); r['wa_abs_err']=float((sub.wa-sub.wa_hat).abs().mean())
    r['bias']=float((sub.eps_hat-sub.eps).mean()); r['rmse']=float(np.sqrt(((sub.eps_hat-sub.eps)**2).mean()))
# ---------- B: fingerprint specificity
def fingerprint(X,names):
    out=[]; n=len(X)
    for j in range(X.shape[1]):
        x=X[:,j]; u=np.unique(x)
        if np.allclose(x,np.round(x)) and len(u)<=200 and len(u)>2:
            lv=np.arange(x.min(),x.max()+1); cnt=np.array([(x==v).sum() for v in lv]); p=stats.chisquare(cnt).pvalue if cnt.min()>=0 and n/len(lv)>=5 else stats.kstest((x-x.min()+np.random.default_rng(1).random(n))/(x.max()-x.min()+1),'uniform').pvalue
        elif len(u)<=2: continue
        else: p=stats.kstest((x-x.min())/(x.max()-x.min()),'uniform').pvalue
        out.append(dict(var=names[j],p_unif=p,kurt=stats.kurtosis(x,fisher=False)))
    C=np.corrcoef(X.T); iu=np.triu_indices(X.shape[1],1); r=np.abs(C[iu]); m=len(r)
    rp=[stats.pearsonr(X[:,i],X[:,j]).pvalue for i,j in zip(*iu)]
    xs=np.abs(X.ravel()); xs=xs[xs>=1]; ld=(xs/10**np.floor(np.log10(xs))).astype(int); o=np.bincount(ld,minlength=10)[1:]
    ben=np.log10(1+1/np.arange(1,10)); chiB=stats.chisquare(o,ben*o.sum())
    df=pd.DataFrame(out)
    return dict(n=n,p=X.shape[1],frac_uniform=float((df.p_unif>0.05).mean()),kurt_med=float(df['kurt'].median()),kurt_range=[float(df['kurt'].min()),float(df['kurt'].max())],
        max_abs_r=float(r.max()),frac_pairs_sig=float((np.array(rp)<0.05/m).mean()),benford_chi_per_value=float(chiB.statistic/o.sum()))
g=pd.read_pickle('d.pkl'); gnum=['Age','PlayTimeHours','SessionsPerWeek','AvgSessionDurationMinutes','PlayerLevel','AchievementsUnlocked']
sets={'Gaming file (audited)':(g[gnum].values.astype(float),gnum)}
bc=load_breast_cancer(); sets['Breast cancer (real, sklearn)']=(bc.data,list(bc.feature_names))
db=load_diabetes(scaled=False); sets['Diabetes (real, sklearn)']=(db.data[:,[0,2,3,4,5,6,7,8,9]],[db.feature_names[i] for i in [0,2,3,4,5,6,7,8,9]])
wn=load_wine(); sets['Wine (real, sklearn)']=(wn.data,list(wn.feature_names))
Xc,_=make_classification(n_samples=40034,n_features=8,n_informative=4,n_redundant=2,random_state=1); sets['make_classification (Gaussian generator)']=(Xc*10+50,[f'x{i}' for i in range(8)])
Xu=rng.uniform(0,100,(40034,6)); sets['Independent uniform (positive control)']=(Xu,[f'u{i}' for i in range(6)])
V['B']={k:fingerprint(X,nm) for k,(X,nm) in sets.items()}
print(json.dumps(V['B'],indent=1),flush=True)
# ---------- C: structure tests detect real clusters
def permcols(X): return np.column_stack([rng.permutation(X[:,j]) for j in range(X.shape[1])])
def knnQ(X):
    nn=NearestNeighbors(n_neighbors=9).fit(X); _,ix=nn.kneighbors(X); G=nx.Graph(); G.add_edges_from((i,j) for i in range(len(X)) for j in ix[i,1:])
    c=nx.community.louvain_communities(G,seed=1); return nx.community.modularity(G,c)
C={}
for name,(X,_) in {'Blobs, 4 clusters (positive control)':(make_blobs(n_samples=3000,n_features=6,centers=4,cluster_std=2.0,random_state=1)[0],None),
                   'Gaming file':(g[gnum].values[rng.choice(len(g),3000,replace=False)].astype(float),None)}.items():
    Z=StandardScaler().fit_transform(X)
    ev=np.linalg.eigvalsh(np.cov(Z.T))[::-1][0]; evn=[np.linalg.eigvalsh(np.cov(permcols(Z).T))[::-1][0] for _ in range(50)]
    bo=[GaussianMixture(k,random_state=1,n_init=2).fit(Z).bic(Z) for k in range(1,7)]; bn=[[GaussianMixture(k,random_state=1,n_init=2).fit(Zn).bic(Zn) for k in range(1,7)] for Zn in [permcols(Z) for _ in range(5)]]
    Qo=knnQ(Z); Qn=[knnQ(permcols(Z)) for _ in range(10)]
    C[name]=dict(eig1=float(ev),eig1_null_hi=float(np.quantile(evn,.975)),bic_best_k=int(np.argmin(bo)+1),bic_gap_best=float(min(bo)-np.mean(bn,0)[np.argmin(bo)]),
        bic_null_range=float(np.ptp(np.array(bn)[:,np.argmin(bo)])),Q=float(Qo),Q_null_max=float(max(Qn)),Q_null_mean=float(np.mean(Qn)))
    print(name,C[name],flush=True)
V['C']=C; json.dump(V,open('out/validation.json','w'),indent=1,default=float); print(json.dumps(V['A'],indent=1,default=float))
