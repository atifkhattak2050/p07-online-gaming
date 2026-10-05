import pandas as pd, numpy as np, json, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
R=json.load(open('out/results.json')); a=pd.read_pickle('d.pkl'); n=len(a)
mpl.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'font.family':'Liberation Sans','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'pdf.fonttype':42})
W=174/25.4
cls=['Low','Medium','High']; col={'Low':'#0072B2','Medium':'#E69F00','High':'#009E73'}; mk={'Low':'o','Medium':'s','High':'^'}
y=a.EngagementLevel.values; S=a.SessionsPerWeek.values; D=a.AvgSessionDurationMinutes.values
def lab(ax,t): ax.text(-0.16,1.06,t,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
MAP={'Figure1':'Fig2','Figure2':'Fig3','Figure3':'Fig4','Figure4':'Fig5','Figure5':'Fig6','Figure6':'Fig7'}
import os; os.makedirs('dm',exist_ok=True)
def save(f,name):
    nm=MAP[name]; f.savefig(f'dm/{nm}.png',dpi=600,bbox_inches='tight'); f.savefig(f'dm/{nm}.eps',bbox_inches='tight'); f.savefig(f'dm/{nm}.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'}); plt.close(f)
rng=np.random.default_rng(1)
# ---------- FIG 1 generative fingerprint
f=plt.figure(figsize=(W,5.6)); gs=f.add_gridspec(2,2,hspace=0.55,wspace=0.32)
g0=gs[0,0].subgridspec(2,3,hspace=0.9,wspace=0.45)
NUM=[('Age','Age (years)'),('PlayTimeHours','Play time (h)'),('SessionsPerWeek','Sessions per week'),('AvgSessionDurationMinutes','Session length (min)'),('PlayerLevel','Player level'),('AchievementsUnlocked','Achievements')]
for i,(c,t) in enumerate(NUM):
    ax=f.add_subplot(g0[i//3,i%3]); x=a[c].values
    if c=='PlayTimeHours': bins=np.linspace(0,24,25)
    else:
        lo,hi=x.min(),x.max(); k=hi-lo+1; step=max(1,int(np.ceil(k/25))); bins=np.arange(lo-0.5,hi+1.5,step)
    cnt,e=np.histogram(x,bins); w=np.diff(e); dens=cnt/n/w
    ax.bar(e[:-1],dens,width=w,align='edge',color='#BBBBBB',edgecolor='none')
    ax.axhline(1/(e[-1]-e[0]),color='k',lw=0.8,ls='--')
    ax.set_title(t,fontsize=6.5,pad=2); ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3)); ax.set_ylim(0,1.6/(e[-1]-e[0])); ax.set_yticks([]); ax.tick_params(labelsize=6)
    if i==0: ax.text(-0.5,1.25,'a',transform=ax.transAxes,fontsize=10,fontweight='bold'); ax.set_ylabel('Density',fontsize=7)
ax=f.add_subplot(gs[0,1]); lab(ax,'b'); B=R['benford']; d=np.arange(1,10)
ax.plot(d,B['benford'],'-',color='#CC79A7',lw=1.2,label='Benford expectation')
ax.plot(d,B['unif'],'--',color='k',lw=1,label='Uniform-generator expectation')
ax.plot(d,B['obs'],'o',ms=4,mfc='white',mec='k',mew=0.9,label=f"Observed (N = {B['N']:,} values)")
ax.set_xticks(d); ax.set_xlabel('Leading digit'); ax.set_ylabel('Proportion of values'); ax.set_ylim(0,0.4)
ax.legend(frameon=False,loc='upper right')
ax.text(0.98,0.55,f"vs Benford: $\\chi^2_8$ = {B['chi_ben']:,.0f}, p < 10$^{{-300}}$\nvs uniform: $\\chi^2_8$ = {B['chi_unif']:.1f}, p = {B['p_unif']:.2f}",transform=ax.transAxes,ha='right',fontsize=6.5)
ax=f.add_subplot(gs[1,0]); lab(ax,'c')
rows=[]
for c,nm in [('Gender','Gender'),('Location','Location'),('GameGenre','Game genre'),('GameDifficulty','Difficulty')]:
    dd=R['dist'][c]
    for lv,o,e in zip(dd['levels'],dd['obs'],dd['exp']): rows.append((f'{nm}: {lv}',o,e))
rows.append(('In-game purchase: yes',R['dist']['InGamePurchases']['prop1'],0.2))
yy=np.arange(len(rows))[::-1]
for yi,(l,o,e) in zip(yy,rows):
    ax.plot([e,e],[yi-0.35,yi+0.35],color='k',lw=1); ax.plot(o,yi,'o',ms=3.5,color='#D55E00')
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows],fontsize=6); ax.set_xlabel('Proportion of records'); ax.set_xlim(0,0.65)
ax.plot([],[],'o',color='#D55E00',ms=3.5,label='Observed'); ax.plot([],[],'-',color='k',label='Round-number probability'); ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=2)
ax=f.add_subplot(gs[1,1]); ax.text(-0.3,1.12,'d',transform=ax.transAxes,fontsize=10,fontweight='bold')
P=np.load('out/pearson.npy'); Dc=np.load('out/dcor.npy'); M=np.full((7,7),np.nan)
il=np.tril_indices(7,-1); iu=np.triu_indices(7,1); M[il]=np.abs(P[il]); M[iu]=Dc[iu]
im=ax.imshow(M,cmap='Greys',vmin=0,vmax=0.1); short=['Age','Play time','Sessions','Duration','Level','Achiev.','Purchase']
ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(short,rotation=45,ha='right',fontsize=6); ax.set_yticklabels(short,fontsize=6)
for i in range(7):
    for j in range(7):
        if i!=j: ax.text(j,i,f'{M[i,j]:.3f}',ha='center',va='center',fontsize=5)
ax.spines[:].set_visible(True)
cb=f.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label('Dependence (0–0.1 scale)',fontsize=6); cb.ax.tick_params(labelsize=6)
pass
save(f,'Figure1')
# ---------- FIG 2 label rule
sc=np.load('out/score.npy'); lf=R['rule']['lin_full']
f,axs=plt.subplots(1,2,figsize=(W,3.0),gridspec_kw=dict(wspace=0.3))
ax=axs[0]; lab(ax,'a'); sub=rng.choice(n,6000,replace=False)
jit=lambda v: v+rng.uniform(-0.3,0.3,len(v))
for c in cls:
    m=sub[y[sub]==c]; ax.scatter(jit(S[m]),D[m],s=2,marker=mk[c],color=col[c],lw=0,label=c)
ss=np.linspace(0.5,19.5,300); mL,mA=a.PlayerLevel.mean(),a.AchievementsUnlocked.mean()
for t,ls in [(lf['t1'],'-'),(lf['t2'],'--')]:
    dcurve=(t-lf['wl']*mL-lf['wa']*mA)/ss; ok=(dcurve>=10)&(dcurve<=180); ax.plot(ss[ok],dcurve[ok],'k',ls=ls,lw=1)
ax.set_xlabel('Sessions per week (S)'); ax.set_ylabel('Average session duration, D (min)'); ax.set_ylim(5,185); ax.set_xlim(-0.8,19.8)
leg=ax.legend(title='Engagement label',frameon=False,markerscale=4,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3,fontsize=6.5,title_fontsize=6.5)
ax=axs[1]; lab(ax,'b')
edges=np.arange(-100,3700,50); mid=(edges[:-1]+edges[1:])/2; b=np.digitize(sc,edges)-1
for c in cls:
    p=[(y[b==i]==c).mean() if (b==i).sum()>=30 else np.nan for i in range(len(mid))]
    nn=np.array([(b==i).sum() for i in range(len(mid))]); p=np.array(p)
    se=np.sqrt(p*(1-p)/np.maximum(nn,1))
    ax.errorbar(mid,p,yerr=1.96*se,fmt=mk[c],ms=2.5,color=col[c],elinewidth=0.5,lw=0,label=c)
for t,ls in [(lf['t1'],'-'),(lf['t2'],'--')]: ax.axvline(t,color='k',ls=ls,lw=0.8)
ax.axhline(R['rule']['agree_far'],color='grey',lw=0.6,ls=':'); ax.axhline(R['rule']['eps']/3,color='grey',lw=0.6,ls=':')
ax.text(3650,R['rule']['agree_far'],f"{R['rule']['agree_far']*100:.1f}%",ha='left',va='center',fontsize=6.5,color='dimgrey',clip_on=False)
ax.text(3650,R['rule']['eps']/3,f"{R['rule']['eps']/3*100:.1f}%",ha='left',va='center',fontsize=6.5,color='dimgrey',clip_on=False)
ax.set_xlabel(f"Recovered engagement score\n(S×D + {lf['wl']}×Level + {lf['wa']}×Achievements)"); ax.set_ylabel('Proportion of records in bin with label')
ax.set_ylim(-0.02,1.02); ax.set_xlim(-100,3600); ax.legend(frameon=False,loc='center right',markerscale=1.5)
save(f,'Figure2')
# ---------- FIG 3 performance & importance
f,axs=plt.subplots(1,2,figsize=(W,3.0),gridspec_kw=dict(wspace=0.6,width_ratios=[1.1,1]))
ax=axs[0]; ax.text(-0.75,1.04,'a',transform=ax.transAxes,fontsize=10,fontweight='bold'); Mo=R['models']
order=['Recovered linear-score rule','RF, all 11 predictors','Threshold rule on S x D','RF, S and D only','LDA, all numeric','Multinomial logistic, S and D','Majority class','RF, without S and D (negative control)']
names={'Recovered linear-score rule':'Recovered score rule (4 parameters)','RF, all 11 predictors':'Random forest, all 11 predictors','Threshold rule on S x D':'Threshold rule on S×D (2 parameters)','RF, S and D only':'Random forest, S and D only','LDA, all numeric':'LDA, all numeric','Multinomial logistic, S and D':'Multinomial logistic, S and D','Majority class':'Majority class (Medium)','RF, without S and D (negative control)':'Random forest without S, D (neg. control)'}
lo,hi=R['rule']['ceiling_ci']; ax.axvspan(lo*100,hi*100,color='#CCCCCC',lw=0); ax.axvline(R['rule']['ceiling']*100,color='k',lw=0.8)
ax.text(R['rule']['ceiling']*100+0.6,len(order)-0.5,f"Estimated\nceiling\n{R['rule']['ceiling']*100:.1f}%",fontsize=6,va='top')
for i,k in enumerate(order[::-1]):
    m_,s_=Mo[k]; ax.errorbar(m_*100,i,xerr=s_*100,fmt='o',ms=3.5,color='k',capsize=2,elinewidth=0.7)
    ax.text(m_*100-1.5,i,f'{m_*100:.1f}',ha='right',va='center',fontsize=6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order[::-1]],fontsize=6.5)
ax.set_xlabel('Five-fold CV accuracy (%), mean ± SD'); ax.set_xlim(40,100)
ax=axs[1]; ax.text(-0.5,1.04,'b',transform=ax.transAxes,fontsize=10,fontweight='bold'); PI=R['perm']['PI']; fs=sorted(PI,key=lambda k:PI[k][0])
nm2={'SessionsPerWeek':'Sessions per week','AvgSessionDurationMinutes':'Session duration','AchievementsUnlocked':'Achievements','PlayerLevel':'Player level','PlayTimeHours':'Play time','InGamePurchases':'In-game purchases','GameDifficulty':'Game difficulty','GameGenre':'Game genre'}
for i,k in enumerate(fs):
    m_,l_,h_=PI[k]; ax.plot([l_*100,h_*100],[i,i],'k',lw=1.2); ax.plot(m_*100,i,'o',ms=3,color='k')
    ax.text(max(h_,0)*100+1,i,f'{m_*100:.2f}',va='center',fontsize=6)
ax.axvline(0,color='grey',lw=0.6); ax.set_yticks(range(len(fs))); ax.set_yticklabels([nm2.get(k,k) for k in fs],fontsize=6.5)
ax.set_xlabel('Accuracy drop when permuted\n(percentage points; mean, 95% CI)'); ax.set_xlim(-2,48)
save(f,'Figure3')
# ---------- FIG 4 structure vs null
f,axs=plt.subplots(1,3,figsize=(W,2.5),gridspec_kw=dict(wspace=0.5))
ax=axs[0]; lab(ax,'a'); G=R['gmm']; k=np.arange(1,7); nulls=np.array(G['null']); nm_=nulls.mean(0)
ax.fill_between(k,nulls.min(0)-nm_,nulls.max(0)-nm_,color='#CCCCCC',lw=0)
ax.plot(k,np.array(G['obs'])-nm_,'o-',color='k',ms=3,lw=0.8); ax.axhline(0,color='grey',lw=0.6,ls=':')
ax.set_xlabel('Gaussian mixture components, k'); ax.set_ylabel('BIC, observed minus null mean')
ax=axs[1]; lab(ax,'b'); Pc=R['pca']; kk=np.arange(1,7)
ax.fill_between(kk,Pc['null_lo'],Pc['null_hi'],color='#CCCCCC',lw=0)
ax.plot(kk,Pc['obs'],'o-',color='k',ms=3,lw=0.8); ax.axhline(1,color='grey',ls=':',lw=0.6)
ax.set_ylim(0.95,1.05); ax.set_xlabel('Principal component'); ax.set_ylabel('Eigenvalue (standardized data)')
ax=axs[2]; lab(ax,'c'); K=R['knn']
ax.hist(K['Qnull'],bins=np.linspace(0.68,0.705,11),color='#CCCCCC',edgecolor='white')
ax.axvline(K['Q'],color='k',lw=1.2); ax.set_xlabel('kNN-graph modularity, Q'); ax.set_ylabel('Null replicates (of 20)'); ax.set_xticks([0.68,0.69,0.70])
from matplotlib.patches import Patch
f.legend([Patch(color='#CCCCCC'),mpl.lines.Line2D([],[],color='k',marker='o',ms=3,lw=0.8)],['Permutation null (a: range of 5 reps; b: 95% band of 50 reps; c: 20 reps)','Observed data'],loc='upper center',bbox_to_anchor=(0.5,1.08),ncol=2,frameon=False,fontsize=6.5)
save(f,'Figure4')
# ---------- FIG 5 collider
f,axs=plt.subplots(1,2,figsize=(W,2.8),gridspec_kw=dict(wspace=0.35,width_ratios=[1,1]))
ax=axs[0]; lab(ax,'a'); m=sub[y[sub]=='Medium']; o=sub[y[sub]!='Medium']
ax.scatter(jit(S[o]),D[o],s=1.5,color='#DDDDDD',lw=0,label='Low or High')
ax.scatter(jit(S[m]),D[m],s=1.5,color=col['Medium'],lw=0,label='Medium only')
z=np.polyfit(S[y=='Medium'],D[y=='Medium'],1); xs=np.array([0,19]); ax.plot(xs,np.polyval(z,xs),'k',lw=1)
ax.set_xlabel('Sessions per week (S)'); ax.set_ylabel('Average session duration, D (min)'); ax.legend(frameon=False,markerscale=5,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2,fontsize=6.5)
ax.text(0.3,8,f"All records: r = {R['berkson']['all']:.3f}\nMedium only: r = {R['berkson']['Medium']['r']:.2f}",fontsize=6.5,va='bottom',bbox=dict(fc='white',ec='none',pad=1.5))
ax.set_ylim(0,185)
ax=axs[1]; ax.text(-0.3,1.06,'b',transform=ax.transAxes,fontsize=10,fontweight='bold'); Bk=R['berkson']; rows=[('All records',Bk['all'],None,None,40034)]+[(f"{c} only",Bk[c]['r'],Bk[c]['null_lo'],Bk[c]['null_hi'],Bk[c]['n']) for c in cls]
for i,(l,r,lo_,hi_,nn) in enumerate(rows[::-1]):
    if lo_ is not None: ax.plot([lo_,hi_],[i,i],color='#999999',lw=5,solid_capstyle='butt')
    ax.plot(r,i,'o',color='k',ms=4)
ax.axvline(0,color='grey',lw=0.6); ax.set_yticks(range(4)); ax.set_yticklabels([f"{r[0]}\n(n = {r[4]:,})" for r in rows[::-1]],fontsize=6.5)
ax.set_xlabel('Pearson r between S and D'); ax.set_xlim(-0.7,0.1)
ax.plot([],[],'o',color='k',label='Observed'); ax.plot([],[],color='#999999',lw=5,label='Null 95% interval (independent S, D;\nrecovered rule + 10% relabelling; 200 reps)')
ax.legend(frameon=False,fontsize=6,loc='lower center',bbox_to_anchor=(0.55,1.02),ncol=2)
save(f,'Figure5')
# ---------- FIG 6 uncertainty
U=R['uq']; f,axs=plt.subplots(1,3,figsize=(W,2.8),gridspec_kw=dict(wspace=0.55))
ax=axs[0]; ax.text(-0.3,1.25,'a',transform=ax.transAxes,fontsize=10,fontweight='bold'); ax.plot([0,1],[0,1],color='grey',lw=0.6,ls=':')
for c in cls:
    for nm,ls,mf in [('raw','--','white'),('iso','-',col[c])]:
        r=np.array(U['rel'][f'{c}_{nm}']); ax.plot(r[:,0],r[:,1],ls=ls,marker=mk[c],ms=3,color=col[c],mfc=mf,lw=0.8)
ax.plot([],[],'k--',marker='o',mfc='white',ms=3,lw=0.8,label=f"Random forest (ECE {U['ece_raw']:.3f})"); ax.plot([],[],'k-',marker='o',ms=3,lw=0.8,label=f"After isotonic (ECE {U['ece_iso']:.3f})")
for c in cls: ax.plot([],[],marker=mk[c],color=col[c],ls='',ms=3,label=c)
ax.set_xlabel('Predicted probability (bin mean)'); ax.set_ylabel('Observed frequency'); ax.legend(frameon=False,fontsize=5.5,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2,columnspacing=0.6)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax=axs[1]; lab(ax,'b'); C=U['conf']; xpos=np.arange(3)
for j,(key,lbl,mf,dx) in enumerate([('marginal_0.1','Standard split conformal','white',-0.12),('mondrian_0.1','Class-conditional (Mondrian)',None,0.12)]):
    for xi,c in enumerate(cls):
        p=C[key]['cc'][c]; k_=C[key]['cc_n'][c]; se=1.96*np.sqrt(p*(1-p)/k_)*100
        ax.errorbar(xi+dx,p*100,yerr=se,fmt=mk[c],ms=4.5,color=col[c],mfc=mf if mf else col[c],mew=1,elinewidth=0.8,capsize=2)
ax.axhline(90,color='k',ls='--',lw=0.8); ax.text(1.5,89.6,'target 90%',fontsize=6,ha='center',va='top')
ax.set_xticks(xpos); ax.set_xticklabels(cls); ax.set_ylim(80,100); ax.set_xlim(-0.5,2.5)
ax.set_yticks([80,85,90,95,100]); ax.set_ylabel('Coverage of true label (%)\nmean, 95% CI (axis from 80%)'); ax.set_xlabel('True engagement class')
ax.plot([],[],'o',mfc='white',mec='k',ms=4,ls='',label='Standard split conformal'); ax.plot([],[],'o',color='k',ms=4,ls='',label='Class-conditional (Mondrian)')
ax.legend(frameon=False,fontsize=6,loc='lower center',bbox_to_anchor=(0.5,1.0))
ax=axs[2]; lab(ax,'c')
keys=[('marginal_0.1','90%\nStd'),('mondrian_0.1','90%\nMond.'),('marginal_0.05','95%\nStd'),('mondrian_0.05','95%\nMond.')]
bottom=np.zeros(4); shades={'1':'#FFFFFF','2':'#999999','3':'#333333'}
for s_ in ['1','2','3']:
    v=np.array([C[k]['dist'][s_]/U['n_test']*100 for k,_ in keys]); ax.bar(range(4),v,0.6,bottom=bottom,color=shades[s_],edgecolor='k',lw=0.5,label=f'{s_} label' + ('s' if s_!='1' else ''))
    bottom+=v
ax.set_xticks(range(4)); ax.set_xticklabels([k[1] for k in keys],fontsize=6); ax.set_ylabel('Test records (%)'); ax.set_xlabel('Target coverage and method')
ax.legend(frameon=False,fontsize=6,loc='upper left',bbox_to_anchor=(0,1.22),ncol=3,columnspacing=0.8,handlelength=1.2); ax.set_ylim(0,100)
save(f,'Figure6')
print('done')
