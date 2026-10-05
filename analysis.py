import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
import networkx as nx
SEED=1; rng=np.random.default_rng(SEED); R={}
a=pd.read_pickle('d.pkl'); n=len(a)
NUM=['Age','PlayTimeHours','SessionsPerWeek','AvgSessionDurationMinutes','PlayerLevel','AchievementsUnlocked']
CAT=['Gender','Location','GameGenre','GameDifficulty']
S=a.SessionsPerWeek.values; D=a.AvgSessionDurationMinutes.values; L=a.PlayerLevel.values; A=a.AchievementsUnlocked.values
cls=['Low','Medium','High']; y=a.EngagementLevel.map({c:i for i,c in enumerate(cls)}).values
# ---- 1 provenance
obs=np.bincount(y,minlength=3); doc=np.array([8648,19374,12012])
R['marg']=dict(obs=obs.tolist(),doc=doc.tolist(),min_relabel=int(np.abs(obs-doc).sum()//2),
  chi=stats.chisquare(obs,doc).statistic, chip=stats.chisquare(obs,doc).pvalue)
# ---- 2 distribution
dist={}
for c in NUM+['InGamePurchases']:
    x=a[c].values
    d=dict(min=float(x.min()),max=float(x.max()),skew=float(stats.skew(x)),kurt=float(stats.kurtosis(x,fisher=False)))
    if c=='PlayTimeHours':
        r=stats.kstest(x,'uniform',args=(0,24)); d.update(test='KS',stat=r.statistic,p=r.pvalue)
    elif c!='InGamePurchases':
        lv=np.arange(x.min(),x.max()+1); cnt=np.array([(x==v).sum() for v in lv])
        r=stats.chisquare(cnt); d.update(test='chi2',stat=r.statistic,df=len(lv)-1,p=r.pvalue,levels=len(lv))
    else:
        r=stats.chisquare([ (x==0).sum(),(x==1).sum()],[0.8*n,0.2*n]); d.update(test='chi2 vs 0.8/0.2',stat=r.statistic,p=r.pvalue,prop1=x.mean())
    dist[c]=d
props={'Gender':{'Male':.6,'Female':.4},'Location':{'USA':.4,'Europe':.3,'Asia':.2,'Other':.1},
 'GameGenre':{k:.2 for k in ['Sports','Action','Strategy','Simulation','RPG']},'GameDifficulty':{'Easy':.5,'Medium':.3,'Hard':.2}}
for c,p in props.items():
    o=np.array([(a[c]==k).sum() for k in p]); e=np.array(list(p.values()))*n
    r=stats.chisquare(o,e); dist[c]=dict(obs=(o/n).round(4).tolist(),exp=list(p.values()),stat=r.statistic,p=r.pvalue,levels=list(p))
R['dist']=dist
# leading digits
def lead(x):
    x=np.abs(x[x>=1]); return (x/10**np.floor(np.log10(x))).astype(int)
ld=np.concatenate([lead(a[c].values.astype(float)) for c in NUM])
obsd=np.bincount(ld,minlength=10)[1:]
ben=np.log10(1+1/np.arange(1,10))
# expectation under uniform generation over each variable's own range (simulation)
sim=[]
for c in NUM:
    x=a[c].values
    if c=='PlayTimeHours': z=rng.uniform(0,24,400000)
    else: z=rng.integers(x.min(),x.max()+1,400000)
    z=z.astype(float); z=z[z>=1]; sim.append(np.bincount(lead(z),minlength=10)[1:]/len(z)*(a[c].values>=1).sum())
unif_exp=np.sum(sim,axis=0)
R['benford']=dict(obs=(obsd/obsd.sum()).tolist(),benford=ben.tolist(),unif=(unif_exp/unif_exp.sum()).tolist(),N=int(obsd.sum()),
  chi_ben=stats.chisquare(obsd,ben*obsd.sum()).statistic,p_ben=stats.chisquare(obsd,ben*obsd.sum()).pvalue,
  chi_unif=stats.chisquare(obsd,unif_exp/unif_exp.sum()*obsd.sum()).statistic,p_unif=stats.chisquare(obsd,unif_exp/unif_exp.sum()*obsd.sum()).pvalue)
fr=np.modf(a.PlayTimeHours.values)[0]; r=stats.kstest(fr,'uniform')
R['frac']=dict(mean=fr.mean(),ks=r.statistic,p=r.pvalue)
R['lag1']={c:float(np.corrcoef(a[c].values[:-1],a[c].values[1:])[0,1]) for c in NUM+['InGamePurchases']}; R['lag1_band']=1.96/np.sqrt(n)
cols=NUM+['InGamePurchases']; C=a[cols].corr().values; R['pearson_maxabs']=float(np.abs(C[np.triu_indices(7,1)]).max())
def dcor(x,z):
    A_=squareform(pdist(x[:,None])); B_=squareform(pdist(z[:,None]))
    A_=A_-A_.mean(0)-A_.mean(1)[:,None]+A_.mean(); B_=B_-B_.mean(0)-B_.mean(1)[:,None]+B_.mean()
    return np.sqrt(max((A_*B_).mean(),0)/np.sqrt((A_*A_).mean()*(B_*B_).mean()))
sub=rng.choice(n,2000,replace=False); X7=a[cols].values[sub].astype(float)
dc=np.zeros((7,7)); 
for i in range(7):
    for j in range(i+1,7): dc[i,j]=dc[j,i]=dcor(X7[:,i],X7[:,j])
nulld=[dcor(X7[:,2],rng.permutation(X7[:,3])) for _ in range(100)]
R['dcor']=dict(max=float(dc[np.triu_indices(7,1)].max()),null95=float(np.quantile(nulld,.95)),n=2000)
np.save('out/pearson.npy',C); np.save('out/dcor.npy',dc)
# ---- 3 label rule
SD=S*D
def best_thr(sc,yy):
    o=np.argsort(sc); ss=sc[o]; yo=yy[o]; m=len(yo)
    c0=np.cumsum(yo==0); c1=np.cumsum(yo==1); c2=np.cumsum(yo==2)
    idx=np.unique(np.linspace(0,m-1,600).astype(int)); best=(0,0,0)
    for i in idx:
        j=idx[idx>i]
        if len(j)==0: continue
        acc=c0[i]+c1[j]-c1[i]+c2[-1]-c2[j]; k=acc.argmax()
        if acc[k]>best[0]: best=(acc[k],ss[i],ss[j[k]])
    return best[0]/m,best[1],best[2]
skf=StratifiedKFold(5,shuffle=True,random_state=SEED); folds=list(skf.split(a,y))
def pred_thr(sc,t1,t2): return np.where(sc>t2,2,np.where(sc>t1,1,0))
accSD=[];accLin=[];W=[]
for tr,te in folds:
    _,t1,t2=best_thr(SD[tr].astype(float),y[tr]); accSD.append((pred_thr(SD[te],t1,t2)==y[te]).mean())
    bb=(0,)
    for wl in [1,1.5,2]:
        for wa in [2.5,3,3.5]:
            sc=(SD+wl*L+wa*A).astype(float); r=best_thr(sc[tr],y[tr])
            if r[0]>bb[0]: bb=(r[0],wl,wa,r[1],r[2])
    W.append(bb[1:]); accLin.append((pred_thr((SD+bb[1]*L+bb[2]*A)[te],bb[3],bb[4])==y[te]).mean())
_,t1,t2=best_thr(SD.astype(float),y); full=(0,)
for wl in np.arange(1,2.51,.25):
    for wa in np.arange(2,4.01,.25):
        r=best_thr((SD+wl*L+wa*A).astype(float),y)
        if r[0]>full[0]: full=(r[0],wl,wa,r[1],r[2])
score=SD+full[1]*L+full[2]*A
far=(np.abs(score-full[3])>150)&(np.abs(score-full[4])>250); pf=pred_thr(score,full[3],full[4])
agree_far=(pf==y)[far].mean()
boot=[ (pf[far][ii]==y[far][ii]).mean() for ii in [rng.integers(0,far.sum(),far.sum()) for _ in range(1000)]]
eps=1.5*(1-agree_far); eps_ci=[1.5*(1-np.quantile(boot,.975)),1.5*(1-np.quantile(boot,.025))]
wrong=(pf!=y)[far]; offd=pd.crosstab(pf[far],y[far],normalize='index')
R['rule']=dict(SD_thr=[float(t1),float(t2)],SD_cv=np.mean(accSD),SD_cv_sd=np.std(accSD),lin_cv=np.mean(accLin),lin_cv_sd=np.std(accLin),
  lin_full=dict(acc=full[0],wl=full[1],wa=full[2],t1=float(full[3]),t2=float(full[4])),folds_w=[list(map(float,w)) for w in W],
  far_n=int(far.sum()),far_frac=far.mean(),agree_far=agree_far,eps=eps,eps_ci=eps_ci,ceiling=1-2*eps/3,
  ceiling_ci=[1-2*eps_ci[1]/3,1-2*eps_ci[0]/3],offdiag=offd.values.round(4).tolist())
# tree on SD only
from sklearn.model_selection import cross_val_score
R['rule']['tree_SD_cv']=cross_val_score(DecisionTreeClassifier(max_depth=6,random_state=SEED),SD[:,None],y,cv=skf).mean()
np.save('out/score.npy',score)
# ---- 4 models
Xall=pd.get_dummies(a.drop(columns=['PlayerID','EngagementLevel']),columns=CAT,drop_first=False).astype(float)
def cv(model,X):
    acc=[]; 
    for tr,te in folds:
        m=model(); m.fit(X.iloc[tr] if hasattr(X,'iloc') else X[tr],y[tr]); acc.append((m.predict(X.iloc[te] if hasattr(X,'iloc') else X[te])==y[te]).mean())
    return float(np.mean(acc)),float(np.std(acc))
rf=lambda: RandomForestClassifier(500,random_state=SEED,n_jobs=-1,min_samples_leaf=5)
import os
if os.path.exists('out/models.json'): M=json.load(open('out/models.json'))
else:
  M={}
  M['RF, all 11 predictors']=cv(rf,Xall)
  M['RF, S and D only']=cv(rf,a[['SessionsPerWeek','AvgSessionDurationMinutes']])
  M['RF, without S and D (negative control)']=cv(rf,Xall.drop(columns=['SessionsPerWeek','AvgSessionDurationMinutes']))
  M['LDA, all numeric']=cv(LinearDiscriminantAnalysis,a[NUM+['InGamePurchases']])
  M['Multinomial logistic, S and D']=cv(lambda: LogisticRegression(max_iter=2000),StandardScaler().fit_transform(a[['SessionsPerWeek','AvgSessionDurationMinutes']]))
  M['Recovered linear-score rule']=(float(np.mean(accLin)),float(np.std(accLin)))
  M['Threshold rule on S x D']=(float(np.mean(accSD)),float(np.std(accSD)))
  M['Majority class']=(float((y==1).mean()),0.0)
  json.dump(M,open('out/models.json','w'))
R['models']=M; print(M,flush=True)
# permutation importance on 25% holdout
tr,te=train_test_split(np.arange(n),test_size=.25,stratify=y,random_state=SEED)
feat=['Age','Gender','Location','GameGenre','PlayTimeHours','InGamePurchases','GameDifficulty','SessionsPerWeek','AvgSessionDurationMinutes','PlayerLevel','AchievementsUnlocked']
def enc(df): return pd.get_dummies(df[feat],columns=CAT).astype(float).reindex(columns=Xall.columns,fill_value=0)
m=rf(); m.fit(enc(a.iloc[tr]),y[tr]); base=(m.predict(enc(a.iloc[te]))==y[te]).mean(); PI={}
for f in feat:
    ds=[]
    for r_ in range(10):
        t=a.iloc[te].copy(); t[f]=rng.permutation(t[f].values); ds.append(base-(m.predict(enc(t))==y[te]).mean())
    ds=np.array(ds); se=ds.std(ddof=1)/np.sqrt(10); PI[f]=[ds.mean(),ds.mean()-1.96*se,ds.mean()+1.96*se]
R['perm']=dict(base=base,PI=PI)
# ---- 5 structure under null
Z=StandardScaler().fit_transform(a[NUM]); sub=rng.choice(n,10000,replace=False); Zs=Z[sub]
def permcols(X): return np.column_stack([rng.permutation(X[:,j]) for j in range(X.shape[1])])
bic_o=[GaussianMixture(k,random_state=SEED,n_init=2).fit(Zs).bic(Zs) for k in range(1,7)]
bic_n=[]
for rep in range(5):
    Zn=permcols(Zs); bic_n.append([GaussianMixture(k,random_state=SEED,n_init=2).fit(Zn).bic(Zn) for k in range(1,7)])
R['gmm']=dict(obs=bic_o,null=np.array(bic_n).tolist())
ev=np.linalg.eigvalsh(np.cov(Z.T))[::-1]; evn=np.array([np.linalg.eigvalsh(np.cov(permcols(Z).T))[::-1] for _ in range(50)])
R['pca']=dict(obs=ev.tolist(),null_lo=np.quantile(evn,.025,0).tolist(),null_hi=np.quantile(evn,.975,0).tolist())
pc1=Z@np.linalg.eigh(np.cov(Z.T))[1][:,-1]; np.save('out/pc1.npy',pc1)
def knnQ(X,lab=None):
    nn=NearestNeighbors(n_neighbors=9).fit(X); _,ix=nn.kneighbors(X)
    G=nx.Graph(); G.add_edges_from((i,j) for i in range(len(X)) for j in ix[i,1:])
    comm=nx.community.louvain_communities(G,seed=SEED); Q=nx.community.modularity(G,comm)
    memb=np.empty(len(X),int)
    for ci,cset in enumerate(comm): memb[list(cset)]=ci
    return Q,memb,len(comm)
s3=rng.choice(n,3000,replace=False); Qo,memb,nc=knnQ(Z[s3]); Qn=[knnQ(permcols(Z[s3]))[0] for _ in range(20)]
ct=pd.crosstab(memb,y[s3]).values; chi=stats.chi2_contingency(ct)[0]; V=np.sqrt(chi/(3000*(min(ct.shape)-1)))
pur=ct.max(1).sum()/3000
R['knn']=dict(Q=Qo,Qnull=Qn,ncomm=nc,V=V,purity=pur,chance=float((y[s3]==1).mean()))
np.save('out/s3.npy',s3); np.save('out/memb.npy',memb)
# ---- 6 Berkson / collider
ber={}
r_all=np.corrcoef(S,D)[0,1]
for k,cn in enumerate(cls):
    msk=y==k; rr=np.corrcoef(S[msk],D[msk])[0,1]
    nl=[]
    for _ in range(200):
        Sp=rng.permutation(S); Dp=rng.permutation(D)  # independent; reapply rule with noise
        sc=Sp*Dp+full[1]*L+full[2]*A; yp=pred_thr(sc,full[3],full[4]); flip=rng.random(n)<eps; yp[flip]=rng.integers(0,3,flip.sum())
        nl.append(np.corrcoef(Sp[yp==k],Dp[yp==k])[0,1])
    ber[cn]=dict(r=rr,n=int(msk.sum()),null_mean=float(np.mean(nl)),null_lo=float(np.quantile(nl,.025)),null_hi=float(np.quantile(nl,.975)))
ber['all']=r_all
top=pc1>=np.quantile(pc1,.7); rp=np.corrcoef(a.InGamePurchases.values[top],S[top])[0,1]
nl=[]
for _ in range(200):
    Zp=permcols(np.column_stack([Z,a.InGamePurchases.values])); p1=Zp[:,:6]@np.linalg.eigh(np.cov(Z.T))[1][:,-1]; t=p1>=np.quantile(p1,.7)
    nl.append(np.corrcoef(Zp[t,6],Zp[t,2])[0,1])
ber['pc1_purch_sess']=dict(r=rp,null_lo=float(np.quantile(nl,.025)),null_hi=float(np.quantile(nl,.975)))
R['berkson']=ber
# ---- 7 calibration & conformal (50/25/25)
idx=rng.permutation(n); itr,ica,ite=idx[:n//2],idx[n//2:3*n//4],idx[3*n//4:]
m=rf(); m.fit(Xall.iloc[itr],y[itr]); Pca=m.predict_proba(Xall.iloc[ica]); Pte=m.predict_proba(Xall.iloc[ite])
iso=[IsotonicRegression(out_of_bounds='clip').fit(Pca[:,k],(y[ica]==k).astype(float)) for k in range(3)]
def cal(P): Q=np.column_stack([iso[k].predict(P[:,k]) for k in range(3)])+1e-9; return Q/Q.sum(1,keepdims=True)
def ece(P,yy,b=10):
    conf=P.max(1); pr=P.argmax(1); e=0
    for lo in np.linspace(0,1,b+1)[:-1]:
        mk=(conf>lo)&(conf<=lo+1/b)
        if mk.any(): e+=mk.mean()*abs((pr[mk]==yy[mk]).mean()-conf[mk].mean())
    return e
# split calibration set into iso-fit half? keep simple: isotonic fit on ica, conformal on ica (note)
Qca=cal(Pca); Qte=cal(Pte)
R['calib']=dict(ece_raw=ece(Pte,y[ite]),ece_iso=ece(Qte,y[ite]),acc=(Pte.argmax(1)==y[ite]).mean())
# reliability per class data
rel={}
for k in range(3):
    for nm,P in [('raw',Pte),('iso',Qte)]:
        bins=np.linspace(0,1,11); b=np.digitize(P[:,k],bins[1:-1])
        rel[f'{cls[k]}_{nm}']=[[float(P[b==i,k].mean()),float((y[ite][b==i]==k).mean()),int((b==i).sum())] for i in range(10) if (b==i).sum()>=20]
R['rel']=rel
alpha=.10
def conformal(Pc,yc,Pt,mondrian=False):
    if not mondrian:
        s=1-Pc[np.arange(len(yc)),yc]; q=np.quantile(s,np.ceil((len(s)+1)*(1-alpha))/len(s),method='higher'); sets=(1-Pt)<=q
    else:
        sets=np.zeros_like(Pt,bool)
        for k in range(3):
            s=1-Pc[yc==k,k]; q=np.quantile(s,np.ceil((len(s)+1)*(1-alpha))/len(s),method='higher'); sets[:,k]=(1-Pt[:,k])<=q
    return sets
out={}
for nm,mond in [('marginal',False),('mondrian',True)]:
    st=conformal(Qca,y[ica],Qte,mond); cov=st[np.arange(len(ite)),y[ite]]; sz=st.sum(1)
    out[nm]=dict(coverage=cov.mean(),cc={cls[k]:float(cov[y[ite]==k].mean()) for k in range(3)},size=sz.mean(),
       dist={int(s):int((sz==s).sum()) for s in range(4)},n=len(ite))
R['conformal']=out
json.dump(R,open('out/results.json','w'),indent=1,default=float)
print(json.dumps({k:R[k] for k in ['marg','rule','perm','gmm','knn','berkson','calib','conformal','benford','frac','dcor','pearson_maxabs']},indent=1,default=float))
