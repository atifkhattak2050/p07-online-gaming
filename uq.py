import pandas as pd, numpy as np, json
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
R=json.load(open('out/results.json')); a=pd.read_pickle('d.pkl'); n=len(a)
cls=['Low','Medium','High']; y=a.EngagementLevel.map({c:i for i,c in enumerate(cls)}).values
CAT=['Gender','Location','GameGenre','GameDifficulty']
# lower-tail ("too good") chi2
for c in ['Age','SessionsPerWeek','AvgSessionDurationMinutes','PlayerLevel','AchievementsUnlocked','InGamePurchases']+CAT:
    d=R['dist'][c]; df=d['df'] if 'df' in d else (len(d['levels'])-1 if isinstance(d.get('levels'),list) else 1)
    d['p_lower']=float(stats.chi2.cdf(d['stat'],df)); d['df']=df
rng=np.random.default_rng(1)
Xall=pd.get_dummies(a.drop(columns=['PlayerID','EngagementLevel']),columns=CAT).astype(float)
idx=rng.permutation(n); itr,iis,ico,ite=idx[:n//2],idx[n//2:5*n//8],idx[5*n//8:3*n//4],idx[3*n//4:]
m=RandomForestClassifier(500,random_state=1,n_jobs=-1,min_samples_leaf=5).fit(Xall.iloc[itr],y[itr])
P={k:m.predict_proba(Xall.iloc[v]) for k,v in dict(iis=iis,ico=ico,ite=ite).items()}
iso=[IsotonicRegression(out_of_bounds='clip').fit(P['iis'][:,k],(y[iis]==k).astype(float)) for k in range(3)]
def cal(Pm): Q=np.column_stack([iso[k].predict(Pm[:,k]) for k in range(3)])+1e-9; return Q/Q.sum(1,keepdims=True)
Qco,Qte=cal(P['ico']),cal(P['ite']); yte=y[ite]
def ece(Pm,yy,b=10):
    conf=Pm.max(1); pr=Pm.argmax(1); e=0
    for lo in np.linspace(0,1,b+1)[:-1]:
        mk=(conf>lo)&(conf<=lo+1/b)
        if mk.any(): e+=mk.mean()*abs((pr[mk]==yy[mk]).mean()-conf[mk].mean())
    return e
rel={}
for k in range(3):
    for nm,Pm in [('raw',P['ite']),('iso',Qte)]:
        b=np.clip((Pm[:,k]*10).astype(int),0,9)
        rel[f'{cls[k]}_{nm}']=[[float(Pm[b==i,k].mean()),float((yte[b==i]==k).mean()),int((b==i).sum())] for i in range(10) if (b==i).sum()>=20]
out=dict(acc=float((P['ite'].argmax(1)==yte).mean()),ece_raw=ece(P['ite'],yte),ece_iso=ece(Qte,yte),n_fit=len(itr),n_iso=len(iis),n_conf=len(ico),n_test=len(ite),rel=rel,conf={})
for alpha in [0.10,0.05]:
    for nm in ['marginal','mondrian']:
        sets=np.zeros_like(Qte,bool)
        groups=[None] if nm=='marginal' else [0,1,2]
        for g in groups:
            msk=np.ones(len(ico),bool) if g is None else (y[ico]==g)
            s=1-Qco[msk,y[ico][msk]]; q=np.quantile(s,min(1,np.ceil((len(s)+1)*(1-alpha))/len(s)),method='higher')
            if g is None: sets=(1-Qte)<=q
            else: sets[:,g]=(1-Qte[:,g])<=q
        cov=sets[np.arange(len(ite)),yte]; sz=sets.sum(1)
        out['conf'][f'{nm}_{alpha}']=dict(cov=float(cov.mean()),cc={cls[k]:float(cov[yte==k].mean()) for k in range(3)},
           cc_n={cls[k]:int((yte==k).sum()) for k in range(3)},size=float(sz.mean()),dist={str(s):int((sz==s).sum()) for s in range(4)})
R['uq']=out; json.dump(R,open('out/results.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if k!='rel'},indent=1))
for c in R['dist']: print(c, R['dist'][c].get('p_lower'))
