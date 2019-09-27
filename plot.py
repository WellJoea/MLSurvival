
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import Ridge, Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
#import scikitplot as skplt
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import fastcluster
import os

def opens(group):
    group = pd.read_csv(group, header=0, encoding='utf-8', sep='\t').fillna(np.nan)

    AYa = group[(group.Group == 'Y') | (group.Group.str.contains('S_')) ].Variables.tolist()
    RYa = group[(group.Group == 'Y') & (group.Type =='R') ].Variables.tolist()
    CYa = group[(group.Group == 'Y') & (group.Type =='C') ].Variables.tolist()
    Sya = group[(group.Group.str.contains('S_')) & (group.Type.isin( ['S', 'T']))]
    Xa  = group[(group.Group != 'Y') & ~(group.Group.str.contains('S_')) ].Variables.tolist()
    Xg  = group[(group.Group != 'Y') & ~(group.Group.str.contains('S_')) ][['Variables','Group']]
    Xg.set_index('Variables', inplace=True)

    SYa = pd.DataFrame([],columns=['T','S']) 
    for i, j in Sya.iterrows():
        SYa.loc[ j.Group , j.Type ] = j.Variables

    return Xg
    #return (group, AYa, RYa, CYa, SYa, Xa, Xg)

class Baseset():
    def __init__(self, outfile, *array, **dicts):
        self.out = outfile
        os.system('mkdir -p '+ os.path.dirname(self.out))

        self.array = array
        self.dicts = dicts
        self.color_ = ['#009E73', '#FF2121', '#00C5CD', '#6600CC', '#E7A72D', '#EE7AE9',
                       '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF', 
                       '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C', '#FB9A99',
                       '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2', '#95A5A6',
                       '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC', '#A473AE',
                       '#FF0000', '#EE7777', '#ED5401']
        self.linestyle_= [ ':', '--', '-.']
        self.markertyle_ = ['X', 'P', '*', 'v', 's','o','^']

        font = {#'family' : 'normal',
                'weight' : 'normal',
                'size'   : 14}
        plt.rc('font', **font)
        plt.figure(figsize=(10,10))
        plt.margins(0,0)
        plt.rcParams.update({'figure.max_open_warning': 1000})
        sns.set(font_scale=1) 
        #plt.rc('xtick', labelsize=20)
        #plt.rc('ytick', labelsize=20)
        #plt.rcParams['font.size'] = 23
        #plt.rcParams.update({'font.size': 22})
        #plt.rcParams['legend.fontsize'] = 'large'
        #plt.rcParams['figure.titlesize'] = 'medium'


class MPlot(Baseset):
    def Feature_Coefs(self, All_coefs_):
        plt.figure(figsize=(13,10))
        color_ = self.color_*All_coefs_.shape[0]
        linestyle_ = self.linestyle_*All_coefs_.shape[0]
        markertyle_ = self.markertyle_*All_coefs_.shape[0]


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10), gridspec_kw={'wspace': 0.1}, sharex=True)
        fig.suptitle('coefs_ importance ROC Accuracy', x=0.5 )
        fig.subplots_adjust(0.05,0.2,0.87,0.95)
        fig.text(0.45, 0.03, '%s featrues'%All_coefs_.shape[0] , ha='center', va='center')
        for x,y,z in zip(['coef_', 'exp(coef_)' ], [ax1, ax2], [ 'coefficients', 'hazard ratio'] ):
            coefs_df = All_coefs_.filter( regex=r'^%s(\.[0-9]+)?$'%x, axis=1 )
            if x == 'exp(coef_)':
                coefs_df = All_coefs_.filter( regex=r'exp\(coef_\)(\.[0-9]+)?$', axis=1 )
            for i,j in enumerate(coefs_df.columns):
                y.plot( coefs_df[j], 
                        marker=markertyle_[i],
                        markersize=3.2,
                        color=color_[i],
                        linestyle=linestyle_[i],
                        lw=1.0, label='')

                y.plot(All_coefs_['%s_mean'%x]  , 'k-.', lw=1.5, label='mean_coef_',alpha= 0.9 )
                y.plot(All_coefs_['%s_median'%x], 'k--', lw=1.0, label='median_coef_',alpha= 0.5 )
                y.fill_between( All_coefs_.index, 
                                All_coefs_['%s_mean'%x]-1*All_coefs_['%s_std'%x], 
                                All_coefs_['%s_mean'%x]+1*All_coefs_['%s_std'%x], 
                                color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')
                if x == 'exp(coef_)':
                    y.plot(All_coefs_['exp(coef_mean)'], 'k:', lw=1.0, label='exp(coef_mean)',alpha= 0.5 )
                y.set_ylabel(x)
                #y.set_xlabel('%s featrues'%All_coefs_.shape[0])
                y.set_xticklabels( All_coefs_.index, rotation='270')

        legend_elements=[ Line2D([0], [0], color=color_[0], marker=markertyle_[0], linestyle=linestyle_[0], markersize=3.2, lw=1, label='CV times'),
                          Line2D([0], [0], color='k', linestyle='-.', lw=1.5, alpha= 0.9, label='coef_/exp(coef_) mean'),
                          Line2D([0], [0], color='k', linestyle='--', lw=1.0, alpha= 0.5, label='coef_/exp(coef_) median'),
                          Line2D([0], [0], color='k', linestyle=':' , lw=1.0, alpha= 0.5, label='exp(coef_mean)'),
                          Patch(facecolor='grey', edgecolor='black' , alpha=0.3, label=r'$\pm$ 1 std. dev.') 
                        ]
        leg = plt.legend(handles=legend_elements, 
                         title='Features', 
                         numpoints=1,
                         bbox_to_anchor=(1.0, 0.5), 
                         prop={'size':11}, loc='center left')
        plt.savefig( self.out, bbox_extra_artists=(leg,) ) # , bbox_inches='tight')
        plt.close()

    def Feature_Import_box(self, All_coefs_, Xg, label, Y_name, Model, sort_by_group=True):
        All_import = All_coefs_.filter( regex=r'^%s.*'%label, axis=1 )

        color_ = self.color_*All_import.shape[0]
        linestyle_ = self.linestyle_*All_import.shape[0]
        markertyle_ = self.markertyle_*All_import.shape[0]

        All_import = pd.concat([All_import, Xg[['Group']]],axis=1, join='inner', sort=False)

        Xg_cor   = All_import.Group.unique()
        cor_dict = dict(zip(Xg_cor, color_[:len(Xg_cor)]))
        All_import['ColorsX'] = All_import.Group.map(cor_dict)

        if sort_by_group:
             All_import.sort_values(by=['Group', label+'_median', label+'_mean'], ascending=[True, False, False], inplace=True, axis=0)

        color_a = ['red' if i >= 0 else 'blue' for i in All_import[label+'_median']]
        color_b = ['red' if i <  0 else 'blue' for i in All_import[label+'_median']]
        color_c  = All_import['ColorsX'].to_list()

        All_raw = All_import[label]
        All_raw = All_import.filter( regex=r'^%s(\.[0-9]+)?$'%label, axis=1 )
        if label == 'exp(coef_)':
            All_raw = All_import.filter( regex=r'exp\(coef_\)(\.[0-9]+)?$', axis=1 )
        print(All_raw.head())

        column  = sorted(set(All_raw.columns))
        X_labels = All_raw.index.to_list()

        Y_sd_min = All_import[label+'_mean'] - 1*All_import[label+'_std']
        Y_sd_max = All_import[label+'_mean'] + 1*All_import[label+'_std']
        #plt.plot(All_import['0_mean'], 'k-.', lw=1.5, label='mean_import', alpha=0.9)
        plt.figure(figsize=(13,10))
        plt.fill_between( X_labels, Y_sd_min, Y_sd_max,
                        color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')

        legend_elements=[ Patch(facecolor=cor_dict[g], edgecolor='r', label=g) for g in sorted(cor_dict.keys()) ]
        if All_import[label+'_median'].min() <0:
            legend_elements.append(Patch(facecolor='white',  edgecolor='blue',  label=r'coefs_ $\geq$0') )
            legend_elements.append(Patch(facecolor='white',  edgecolor='red' ,  label=r'coefs_ <0') )
        legend_elements.append(Patch(facecolor='grey', edgecolor='black' , alpha=0.3, label=r'$\pm$ 1 std. dev.') )
        ncol_ = 1 if len(legend_elements) <=6 else 2

        bplot =plt.boxplot(All_raw,
                        patch_artist=True,
                        vert=True,
                        labels=X_labels,
                        notch=0,
                        positions=range(len(X_labels)),
                        meanline=True,
                        showmeans=True,
                        meanprops={'linestyle':'-.'}, #'marker':'*'},
                        sym='+',
                        whis=1.5
                        )
        for i, patch in enumerate(bplot['boxes']):
            patch.set(color=color_b[i], linewidth=1.3)
            patch.set(facecolor = color_c[i])

        for element in [ 'means','medians','fliers']:     #['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
            for i, patch in enumerate(bplot[element]):
                patch.set(color=color_b[i], linewidth=1)

        plt.title(Y_name + ' ' + Model + '  coefs_ ROC Accuracy')
        plt.legend(handles=legend_elements, ncol=ncol_, prop={'size':11}, loc='upper right')
        plt.ylabel(Y_name + ' ' + Model + ' coefs_ values')
        plt.xticks(rotation='270')
        plt.savefig(self.out, bbox_inches='tight')

    def Score_1(self, score ):
        All_mean   = score.groupby([score.index]).mean()
        All_median = score.groupby([score.index]).median()
        All_Score  = All_mean[['OS_months', 'OS_status_B']]
        All_Score['risk_score_mean']   = All_mean['risk_score']
        All_Score['risk_score_median'] = All_median['risk_score']
        All_Score.sort_values(by=['OS_months', 'OS_status_B'],
                             ascending=[True, True], inplace=True, axis=0)
        All_Score.to_csv('bb.xls',sep='\t',index=True)

        #fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(All_Score.shape[0]/5,13), gridspec_kw={'wspace': 0, 'hspace': 0}, sharex=True)
        plt.figure(figsize=(All_Score.shape[0]/5,13))
        #fig.suptitle('coefs_ importance ROC Accuracy', x=0.5 )
        #fig.subplots_adjust(0.05,0.2,0.87,0.95)

        ax1 = plt.subplot2grid((13, 1), (0, 0), rowspan=10)
        ax1.plot(All_Score['risk_score_mean'],   'r--', marker='*', linewidth=1.3, markersize=4.2, label='risk_score_mean')
        ax1.plot(All_Score['risk_score_median'], 'b:' , marker='^', linewidth=1.3, markersize=4.2, label='risk_score_median')
        ax1.scatter(score.index, score['risk_score'], c='g', marker='o', s=15.2)
        ax1.set_xticks([])

        ax2 = plt.subplot2grid((13, 1), (10, 0), rowspan=2)
        ax2.plot(All_Score['OS_months'], 'k-' , marker='s', linewidth=1.5, markersize=4.2, label='risk_score_median')
        #ax1.scatter(score.index, score['risk_score'], c='g', marker='o', s=15.2)

        ax3 = plt.subplot2grid((13, 1), (12, 0), rowspan=1)
        ax3.pcolor(All_Score[['OS_status_B']].T, cmap=plt.cm.summer)
        ax3.set_yticks([])
        ax3.set_ylabel('OS_status')
        #ax3.legend()
        plt.xticks(rotation='270')

        plt.savefig( self.out ) #,bbox_inches='tight')
        plt.close()

    def Score_(self, score ):
        
        All_mean   = score.groupby([score.index]).mean()
        All_median = score.groupby([score.index]).median()
        All_Score  = All_mean[['OS_months', 'OS_status_B']]
        All_Score['risk_score_mean']   = All_mean['risk_score']
        All_Score['risk_score_median'] = All_median['risk_score']
        All_Score.sort_values(by=['OS_months', 'OS_status_B'],
                             ascending=[True, True], inplace=True, axis=0)
        All_Score.to_csv('bb.xls',sep='\t',index=True)

        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(All_Score.shape[0]/5,13) )
        fig.suptitle('coefs_ importance ROC Accuracy', x=0.5,y =0.92 )
        gs  = gridspec.GridSpec(13, 1)

        ax1 = plt.subplot(gs[:1, :])
        ax1.pcolor(All_Score[['OS_status_B']].T, cmap=plt.cm.summer)
        ax1.set_xticks([])
        ax1.set_ylabel('OS_status')

        ax2 = plt.subplot(gs[1:3, :])
        ax2.grid(True)
        ax2.plot(All_Score['OS_months'], 'c-' , marker='s', linewidth=1.5, markersize=4.2, label='risk_score_median')
        ax2.pcolor(All_Score[['OS_status_B']].T, cmap=plt.cm.winter)
        ax2.set_xticks([])
        ax2.set_ylabel('OS_months')
        ax2.grid(which='major', axis='both')

        ax3 = plt.subplot(gs[3:14, :] )
        ax3.plot(All_Score['risk_score_mean'], 'r--', marker='*', linewidth=1.3, markersize=4.2, label='risk_score_mean')
        ax3.plot(All_Score['risk_score_median'], 'b:' , marker='^', linewidth=1.3, markersize=4.2, label='risk_score_median')
        ax3.scatter(score.index, score['risk_score'], c='g', marker='o', s=15.2, label='risk_score_all')
        ax3.set_xticklabels( All_Score.index, rotation='270')
        ax3.set_ylabel('risk_scores')
        ax3.set_xlim(All_Score.index[0], All_Score.index[-1])
        ax3.legend()

        plt.savefig( self.out ,bbox_inches='tight')
        plt.close()

def scor_S(score):
    All_mean   = score.groupby([score.index]).mean()
    All_median = score.groupby([score.index]).median()
    All_Score  = All_mean[['OS_months', 'OS_status_B']]
    All_Score['risk_score_mean']   = All_mean['risk_score']
    All_Score['risk_score_median'] = All_median['risk_score']
    All_Score['exp_risk_score_mean']   = np.exp(All_mean['risk_score'])
    All_Score['exp_risk_score_median'] = np.exp(All_median['risk_score'])

    All_Score.to_csv('aa.xls',sep='\t',index=True)


#file = r'C:\Users\lenovo\Desktop\MLSurvival\MLtest\Result\CoxPH\03ModelFit\CoxPH_S_OS_months_Survival_coefficients.xls'
#dataf= pd.read_csv(file, header=0, index_col='Sample', encoding='utf-8', sep='\t').fillna(np.nan)
#print(dataf.head())

#group=r'C:\Users\lenovo\Desktop\MLSurvival\MLtest\Data.222.25.group.txt'
#group=r'~/Desktop/MLSurvival/MLtest/Data.222.25.group.txt'

#Xg = opens(group)
#print(Xg.head())
#MPlot('aa.pdf').Feature_Import_box(dataf, Xg, 'coef_', 'OS_months', 'CoxPH', sort_by_group=False)

#score=r'C:\Users\lenovo\Desktop\MLSurvival\MLtest\Result\CoxPH\03ModelFit\CoxPH_S_OS_months_Survival_TrainTest_risk_score.detail.xls'
#score=r'~/Desktop/MLSurvival/MLtest/Result/CoxPH/03ModelFit/CoxPH_S_OS_months_Survival_risk_score.xls'
#MPlot('aa.pdf').Score_(score)

#score=r'C:\Users\lenovo\Desktop\MLSurvival\MLtest\Result\CoxPH\03ModelFit\CoxPH_S_RFS_months_Survival_TrainTest_risk_score.final.xls'
#score=r'~/Desktop/MLSurvival/MLtest/Result/CoxPH/03ModelFit/CoxPH_S_OS_months_Survival_risk_score.xls'
#score= pd.read_csv(score, header=0, index_col='Sample', encoding='utf-8', sep='\t').fillna(np.nan)
#print(score)

from sksurv.datasets import load_veterans_lung_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

def NotParamete(df, event, time, term):
    cph = CoxPHFitter()
    cheakpoint = np.unique( np.percentile(df[term] , np.linspace(10, 90, 99)) )
    for ic in cheakpoint:
        point = (df[term] >= ic)
        T1 = df[point][time]
        E1 = df[point][event]
        T2 = df[~point][time]
        E2 = df[~point][event]
        results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)

        cph.fit(df[[event, time, term]], time, event)
        cph.print_summary()
        print(cph.hazard_ratios_, cph.score_)
        print(results.p_value)

    return(results)

def COX():
    estimator = CoxPHSurvivalAnalysis(alpha=0.5, n_iter=100, tol=1e-09, verbose=0)
    parameters = {  'alpha'  : [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    'n_iter' : [300],
                    'tol'    : [1e-09],
                    'verbose': [0] },
    return ( estimator, parameters )
def NET():
    estimator = CoxnetSurvivalAnalysis(n_alphas=1000,
                                        alphas=None,
                                        alpha_min_ratio=0.0001,
                                        l1_ratio=0.5,
                                        penalty_factor=None,
                                        normalize=False,
                                        copy_X=True,
                                        tol=1e-07,
                                        max_iter=100000,
                                        verbose=False,
                                        fit_baseline_model=False,
                                        )

    parameters = {  'n_alphas'          : [ 50, 80, 100, 200, 500, 700, 1000 ] ,
                    'alphas'            : [None],
                    'alpha_min_ratio'   : [0.0001],
                    'l1_ratio'          : np.arange(0.01,1,0.01),
                    'penalty_factor'    : [None],
                    'normalize'         : [False],
                    'copy_X'            : [True],
                    'tol'               : [1e-07],
                    'max_iter'          : [100000],
                    'verbose'           : [False],
                    'fit_baseline_model': [False],
                 }
    return ( estimator, parameters )


#Data=r'C:\Users\lenovo\Desktop\MLSurvivalv0.01\test\Result\CoxPH\02FeatureSLT\CoxPH_S_OS_months_Survival_FeatureSLT.Data.xls'
Data = r'./test/Result/CoxPH/02FeatureSLT/CoxPH_S_OS_months_Survival_FeatureSLT.Data.xls'
DF = pd.read_csv(Data, header=0, index_col='Sample', encoding='utf-8', sep='\t').fillna(np.nan)

_S_Time= 'OS_status'
_T_Time= 'OS_months'
_Xa = [i for i in DF.columns if i not in [_S_Time, _T_Time]]
DF[_S_Time +'_B'] = DF[_S_Time].apply(lambda x: True if x >0 else False)
Y_df = DF[[_S_Time+'_B', _T_Time ]].to_records(index = False)
print(DF.describe())

cph = CoxPHFitter(alpha=0.05, tie_method='Efron', penalizer=0.5, strata=None)
cph.fit( DF[ _Xa + [_S_Time, _T_Time ]], _T_Time, _S_Time )
cph.print_summary()
print(cph.score_)

def COX_(cox):
    cphe, cphp = cox
    cphe.fit(DF[_Xa], Y_df)
    print(cphe.coef_, np.exp(cphe.coef_), cphe.score(DF[_Xa], Y_df))
#COX_(COX())


import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from sklearn.model_selection import (StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, 
                                     RepeatedKFold, RepeatedStratifiedKFold, train_test_split,
                                     GridSearchCV, RandomizedSearchCV )
                                
from sklearn.utils import safe_sqr
def NET_(net):
    cphe, cphp = net
    cphe.fit(DF[_Xa], Y_df)
    print(cphe.coef_.shape, cphe.alphas_,1111) # np.exp(cphe.coef_), cphe.score(DF[_Xa], Y_df))
    print(cphe.coef_[:, -1], 2222)
    print(cphe.predict(DF[_Xa]),999999999999999999)
    #print(cphe.coef_)
    #plt.plot(cphe.alphas_, safe_sqr(cphe.coef_).T)
    #plt.plot(cphe.alphas_,cphe.deviance_ratio_)
    aa = cphe.alphas_
    bb = cphe.coef_.T 

    scorcs= []
    print(cphe)
    print(cphe.score(DF[_Xa], Y_df),22222)
    for i in aa:
        cphe.set_params(alphas= [i])
        cphe.fit(DF[_Xa], Y_df)
        scorcs.append(cphe.score(DF[_Xa], Y_df))
    #print( aa,  scorcs)
    #plt.plot(aa,scorcs)
    #plt.show()

    scorcs = np.array(scorcs)
    print(np.where(scorcs==scorcs.max()), scorcs[np.where(scorcs==scorcs.max())], 3333333333)
    aa_best= aa[np.where(scorcs==scorcs.max())]

    cphe.set_params(alphas= aa_best)
    cphe.fit(DF[_Xa], Y_df)
    print( cphe.score(DF[_Xa], Y_df) ,55555555555) 
    print( cphe.coef_, 6666666666)
    print( cphe.predict(DF[_Xa]),999999999999999999)
    return (aa, bb )
cbs_, ccf_ = NET_(NET())

def SNet_():
    class CoxnetSurvivalAnalysis_(CoxnetSurvivalAnalysis):
        #def __init__(self,  **kwargs):
        #    #super().__init__()
        #    CoxnetSurvivalAnalysis.__init__(self)
        #    #super(CoxnetSurvivalAnalysis, self).__init__()
        #    self.coefs_ = ''

        def fit(self, X, y):
            super(CoxnetSurvivalAnalysis_, self).fit(X, y) 
            self.max_id = np.where( self.deviance_ratio_ == self.deviance_ratio_.max() )
            self.coefs_ = self.coef_ 
            self.coef_  = self.coefs_[:, self.max_id[-1] ]
            self.alphab_ = self.alphas_[self.max_id[-1]]
            return self

        def _get_coef(self, alpha):
            check_is_fitted(self, "coef_")

            if alpha is None:
                coef = self.coef_[:, -1]
            else:
                coef = self._interpolate_coefficients(alpha)
            return coef

        def predict(self, X):
            super(CoxnetSurvivalAnalysis_, self).predict(X, alpha=self.alphab_) 
    
    cphe  = CoxnetSurvivalAnalysis_( n_alphas=1000,
                                    alphas=None,
                                    alpha_min_ratio=0.0001,
                                    l1_ratio=0.5,
                                    penalty_factor=None,
                                    normalize=False,
                                    copy_X=True,
                                    tol=1e-07,
                                    max_iter=100000,
                                    verbose=False,
                                    fit_baseline_model=False,
                                    )
    cphe.fit(DF[_Xa], Y_df)
    print( cphe.score(DF[_Xa], Y_df ) ,55555555555) 
    print(cphe.max_id, 23232323, cphe.coef_, cphe.coefs_.shape) # np.exp(cphe.coef_), cphe.score(DF[_Xa], Y_df))
    print(cphe.predict(DF[_Xa], alpha = cphe.max_id),9999999999999)
    return (cphe.alphas_, cphe.coefs_.T )
als_ , cs_ = SNet_()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10), gridspec_kw={'wspace': 0.1} ) #, sharex=True)
ax1.plot(cbs_, ccf_ )
ax2.plot(als_ , cs_ )
plt.show()


def Grid(NET):
    estimator, parameters =  NET
    clf = GridSearchCV( estimator, parameters,
                        n_jobs=-1, 
                        cv=5,
                        scoring=None,
                        error_score = np.nan,
                        return_train_score=True,
                        refit = True,
                        iid=True)

    clf.fit(DF[_Xa], Y_df)
    scor_ = clf.score(DF[_Xa], Y_df)
    print(clf.best_params_)
    print(clf.best_estimator_.coef_.shape, clf.best_estimator_.alphas_) # np.exp(cphe.coef_), cphe.score(DF[_Xa], Y_df))
    #print(clf.predict(DF[_Xa]))
    print(scor_, 1111111111)

    aa = clf.best_estimator_.alphas_


    '''
    from copy import copy
    es = copy( clf.best_estimator_ )
    ps = {"alphas": [[v] for v in aa]}
    glf = GridSearchCV( es , ps ,
                    n_jobs=-1, 
                    cv=5,
                    scoring=None,
                    error_score = np.nan,
                    return_train_score=True,
                    refit = True,
                    iid=True)
    glf.fit(DF[_Xa], Y_df)

    print(glf.score(DF[_Xa], Y_df), 2222222222)
    print(glf.best_estimator_.alphas_, 2222222222)
    '''

    plt.plot(aa, clf.best_estimator_.deviance_ratio_)
    scorcs= []
    for i in aa:
        clfa = clf.best_estimator_
        clfa.set_params(alphas= [i])
        clfa.fit(DF[_Xa], Y_df)
        scorcs.append(clfa.score(DF[_Xa], Y_df))
    print(len( scorcs) , len(aa), 44444)
    scorcs = np.array(scorcs)
    print(np.where(scorcs==scor_), scorcs[np.where(scorcs==scor_)], 222222222)
    print(np.where(scorcs==scorcs.max()), scorcs[np.where(scorcs==scorcs.max())], 3333333333)

    aa_best= aa[np.where(scorcs==scorcs.max())]

    clfn = clf.best_estimator_
    clfn.set_params(alphas= aa_best)
    clfn.fit(DF[_Xa], Y_df)
    print( clfn.score(DF[_Xa], Y_df) ,55555555555) 
    print( clfn.coef_, 6666666666)


    parameters_n = parameters
    parameters_n['alphas'] = [ aa_best]
    print(parameters_n)

    clf_p = GridSearchCV( clf.best_estimator_, parameters_n,
                        n_jobs=-1, 
                        cv=5,
                        scoring=None,
                        error_score = np.nan,
                        return_train_score=True,
                        refit = True,
                        iid=True)

    clf_p.fit(DF[_Xa], Y_df)
    scor_p = clf_p.score(DF[_Xa], Y_df)
    print(scor_p)

    '''
    plt.plot(aa,scorcs, label='each')
    #plt.plot(aa,scor_, label='all')
    plt.xlim([0.01,0.020])
    plt.ylim([0.78,0.82])
    plt.show()
    '''

#Grid(NET())