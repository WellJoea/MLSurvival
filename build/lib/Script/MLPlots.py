from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import Ridge, Lasso
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis

from statsmodels.stats.outliers_influence import variance_inflation_factor
#import scikitplot as skplt
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
elif platform.system()=='Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import fastcluster
import os

class Baseset():
    def __init__(self, outfile, *array, **dicts):
        self.out = outfile
        os.makedirs( os.path.dirname(self.out), exist_ok=True)

        self.array = array
        self.dicts = dicts
        self.color_ = [ '#00BD89', '#DA3B95', '#16E6FF', '#E7A72D', '#8B00CC', '#EE7AE9',
                        '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
                        '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
                        '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
                        '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
                        '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]

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

class ClusT(Baseset):
    def Plot_person(self, dfa, Xa, Xg):
        VIF_df = dfa[Xa].assign(const=1)
        VIF    = pd.Series( [ variance_inflation_factor(VIF_df.values, i) 
                              for i in range(VIF_df.shape[1])
                            ], index=VIF_df.columns)

        cr = np.corrcoef(dfa[Xa].values,rowvar=False)
        cf = pd.DataFrame(cr, index=Xa,columns=Xa)

        cf['VIF'] = VIF.drop('const')
        Xg_cor = Xg.Group.unique()
        cor_dict = dict(zip(Xg_cor, self.color_[:len(Xg_cor)]))
        cf['Group'] = Xg.Group.map(cor_dict)

        vif_dict = {}
        for i in cf.VIF.unique():
            if i <= 5:
                vif_dict[i] = plt.cm.Greens(i/10) 
            elif 5 <= i <10:
                vif_dict[i] = plt.cm.Blues(i/10)
            elif i >= 10:
                vif_dict[i] = plt.cm.Reds(i/cf.VIF.max())
        cf['VIFs'] = cf.VIF.map(vif_dict)

        cf.to_csv( self.out+'.xls', sep='\t' )

        linewidths= 0 if min(cr.shape) > 60  else 0.01
        figsize   = (20,20) if min(cr.shape) > 60  else (15,15)

        hm = sns.clustermap(cf[Xa],
                            method='complete',
                            metric='euclidean',
                            z_score=None,
                            figsize=figsize,
                            linewidths=linewidths,
                            cmap="coolwarm",
                            center=0,
                            #fmt='.2f',
                            #square=True, 
                            #cbar=True,
                            #yticklabels=Xa,
                            #xticklabels=Xa,
                            vmin=-1.1,
                            vmax=1.1,
                            annot=False,
                            row_colors=cf[['Group','VIFs']],
                            col_colors=cf[['Group','VIFs']],
                            )
        hm.savefig(self.out)
        #hm.fig.subplots_adjust(right=.2, top=.3, bottom=.2)
        plt.close()

    def Plot_pair(self, dfX):
        if dfX.shape <= (15, 15):
            #figsize   = (30,30) if min(dfX.shape) > 50  else (25,25)
            #plt.figure(figsize=figsize)
            #hm = sns.pairplot(dfX, height=6, plot_kws ={'edgecolor' : None},kind="reg")
            sns.set_style("whitegrid", {'axes.grid' : False})
            hm = sns.pairplot(dfX,height=6)
            hm.savefig(self.out)
            #plt.figure(figsize=(10,10))
            plt.close()

    def Plot_hist(self, dfaA, dfa, Xa):
        with PdfPages(self.out) as pdf:
            for Xi in Xa:
                plt.figure()
                ax1 = plt.subplot(1,2,1)
                ax2 = plt.subplot(1,2,2)
                plt.sca(ax1)
                sns.distplot(dfa[Xi],bins=100, label='before scale', hist=True, kde=True, rug=True,
                            rug_kws={"color": "g"},
                            kde_kws={"color": "b", "lw": 1.1, "label": "gaussian kernel"},
                            hist_kws={"alpha": 1,'lw':0.01, "color": "r"}
                            )
                plt.sca(ax2)
                sns.distplot(dfaA[Xi],bins=100, label='after scale', hist=True, kde=True, rug=True,
                            rug_kws={"color": "g"},
                            kde_kws={"color": "b", "lw": 1.1, "label": "gaussian kernel"},
                            hist_kws={"alpha": 1, 'lw':0.01, "color": "r"}
                            )
                pdf.savefig()
                plt.close()

    def Plot_heat(self, Xdf, Ydf, Xg, median=None, method='complete', metric='euclidean' ):
        cm_r1 = [plt.cm.YlOrBr, plt.cm.RdPu, plt.cm.YlGnBu ]
        cm_r2 = [plt.cm.bwr, plt.cm.coolwarm, plt.cm.RdYlGn]

        if isinstance(Xdf, pd.DataFrame) & (len(Xdf.shape) >1):
            Xdf = Xdf.fillna(0)
            _Xa = Xdf.columns.tolist()
            row_dt = None
            col_dt = None

            if len(Xg) >0:
                Xg_cor = Xg.Group.unique()
                cor_dict = dict(zip(Xg_cor, self.color_[:len(Xg_cor)]))
                Xg['Colors'] = Xg.Group.map(cor_dict)
                Xg  = Xg.loc[_Xa]
                col_dt = Xg.Colors.values

            if len(Ydf) >0:
                if len(Ydf.shape)==1:
                    Ydf = Ydf.to_frame()
                row_dt = Ydf.copy()

                n1=0
                n2=0
                for i in Ydf.columns:
                    _cors = sorted(Ydf[i].unique())
                    if len(_cors) <=8:
                        cor_dict = dict(zip(_cors,  self.color_[:len(_cors)]))
                    else:
                        (min_, max_) = ( Ydf[i].min(), Ydf[i].max())
                        if median != None:
                            cmap = cm_r2[n2]
                            median_sd = max(max_- median, median-min_ )
                            max_n, min_n = median- median_sd, median + median_sd
                            colormap  = [ cmap( (i-min_n)*0.9/(max_n-min_n) )  for i in _cors ]
                            n2 += 1
                        else:
                            cmap = cm_r1[n1]
                            colormap  = [ cmap( (i-min_)*0.9/(max_-min_) )  for i in _cors ]
                            n1 += 1
                        cor_dict = dict(zip( _cors, colormap ))
                    row_dt[i] = Ydf[i].map(cor_dict)

            linewidths= 0 if max(Xdf.shape) > 60  else 0.01
            figsize   = (25,25) if max(Xdf.shape) > 60 else (18,18)
            cmap = None if Xdf.values.min() >=0 else 'coolwarm'

            hm = sns.clustermap(Xdf,
                                method=method,
                                metric=metric,
                                z_score=None,
                                figsize=figsize,
                                linewidths=linewidths,
                                cmap=cmap,
                                #yticklabels=_Xa,
                                #xticklabels=_AllDF.index.tolist(),
                                annot=False,
                                row_colors=row_dt,
                                col_colors=col_dt
                                )
            hm.savefig(self.out)
            plt.close()

    def Collinearsk(self, Xdf, Ydf):
        color_ = ( self.color_*Xdf.shape[1] )[: Xdf.shape[1] ]
        linestyle_ = (self.linestyle_*Xdf.shape[1] )[: Xdf.shape[1] ]
        markertyle_ = (self.markertyle_*Xdf.shape[1] )[: Xdf.shape[1] ]

        def coefs_sk(mode):
            n_alphas = 100
            alphas = np.logspace(-4,6,num=n_alphas, endpoint=True)
            coefs  = []
            for a in alphas:
                clf = mode(alpha=a, fit_intercept=False)
                clf.fit(Xdf, Ydf)
                coefs.append(clf.coef_)
            coefs = pd.DataFrame(coefs, columns=Xdf.columns, index= alphas)
            return coefs

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10), gridspec_kw={'wspace': 0.1}, sharex=True)
        for x,y,z in zip([Ridge, Lasso], [ax1, ax2], ['Ridge', 'Lasso'] ):
            coefs_df = coefs_sk(x)
            coefs_df.to_csv( '%s.%s.xls'%(self.out, z), sep='\t' )
            for i,j in enumerate(coefs_df.columns):
                y.plot( coefs_df[j], 
                        marker=markertyle_[i],
                        markersize=3.2,
                        color=color_[i],
                        linestyle=linestyle_[i],
                        lw=1.0, label='')
                y.set_xscale('log')
                y.set_title(z + ' coefs')
                y.set_ylabel('weights')
                y.set_xlabel('alpha')

        #plt.margins(x=0)
        leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1, labels=Xdf.columns, title='Features')
        plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.close()

class MPlot(Baseset):
    def Feature_Sorce(self, select_Sco, select_Fea, N, Y, cv):
        color_ = self.color_*select_Sco.shape[1]
        linestyle_ = self.linestyle_*select_Sco.shape[1]
        markertyle_ = self.markertyle_*select_Sco.shape[1]

        Sco_mean = select_Sco.mean(axis=1)
        Sco_Std  = select_Sco.std(axis=1)
        Fea_mean = select_Fea.mean()
        Fea_Std  = select_Fea.std()
        Fea_min  = (Fea_mean-1*Fea_Std) if (Fea_mean-1*Fea_Std) > 0 else 1
        Fea_max  = (Fea_mean + 1*Fea_Std) if (Fea_mean + 1*Fea_Std)<= select_Sco.shape[0] else select_Sco.shape[0]
        for i in range(select_Sco.shape[1]):
            plt.plot(select_Sco[i], marker=markertyle_[i], markersize=3.2, color=color_[i], linestyle=linestyle_[i], lw=1.0, label='')
            plt.vlines(select_Fea[i], 0, 1, color=color_[i], linestyle=linestyle_[i], lw=1.2, label='')
        plt.plot(Sco_mean , 'k-', lw=1.5, label='mean_score')
        plt.vlines(Fea_mean, 0, 1, lw=1.5, label='')
        plt.fill_between(Sco_mean.index, Sco_mean-1*Sco_Std, Sco_mean + 1*Sco_Std, color='grey', alpha=0.25, label=r'$\pm$ 1 std. dev.')
        plt.axvspan(Fea_min, Fea_max , alpha=0.3, color='grey', label='')

        plt.title('The %s features score of %s with %s estimator'%(cv, Y, N))
        plt.legend(loc='lower right')
        plt.ylabel('The %s features score'%cv)
        plt.xlabel('n featrues')
        plt.ylim([0.0, 1.0])
        plt.savefig( self.out )
        plt.close()

    def Feature_Coefs(self, All_coefs_, _T_name , model):
        plt.figure(figsize=(13,10))
        color_ = self.color_*All_coefs_.shape[0]
        linestyle_ = self.linestyle_*All_coefs_.shape[0]
        markertyle_ = self.markertyle_*All_coefs_.shape[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10), gridspec_kw={'wspace': 0.1}, sharex=True)
        fig.suptitle('The features coef/exp_coef_ of %s with %s estimator'%(_T_name, model ), x=0.5 )
        fig.subplots_adjust(0.05,0.2,0.87,0.95)
        fig.text(0.45, 0.03, 'The %s featrues'%All_coefs_.shape[0] , ha='center', va='center')
        for x,y,z in zip(['coef_', 'exp_coef_' ], [ax1, ax2], [ 'coefficients', 'hazard ratio'] ):
            coefs_df = All_coefs_[x]
            for i in range(coefs_df.shape[1]):
                y.plot( coefs_df.iloc[:,i], 
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
                            color='grey', alpha=0.25, label=r'$\pm$ 1 std. dev.')
            if x == 'exp_coef_':
                y.plot(All_coefs_['exp(coef_mean)'], 'k:', lw=1.0, label='exp(coef_mean)',alpha= 0.5 )
            y.set_ylabel(x + ' / ' + z)
            y.set_xticklabels( All_coefs_.index, rotation='270')

        legend_elements=[ Line2D([0], [0], color=color_[0], marker=markertyle_[0], linestyle=linestyle_[0], markersize=3.2, lw=1, label='CV times'),
                          Line2D([0], [0], color='k', linestyle='-.', lw=1.5, alpha= 0.9, label='coef_/exp_coef_ mean'),
                          Line2D([0], [0], color='k', linestyle='--', lw=1.0, alpha= 0.5, label='coef_/exp_coef_ median'),
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

    def Feature_Coefs_box(self, All_coefs_, Xg, label, _T_name, model, sort_by_group=True):
        All_import = All_coefs_.filter( regex=r'^%s.*'%label, axis=1 )
        color_ = self.color_*All_import.shape[0]
        linestyle_ = self.linestyle_*All_import.shape[0]
        markertyle_ = self.markertyle_*All_import.shape[0]

        All_import = pd.concat([All_import, Xg[['Group']]],axis=1, join='inner', sort=False)
        All_import.sort_values(by=[label+'_mean', label+'_median'], ascending=[False, False], inplace=True, axis=0)
        if sort_by_group:
             All_import.sort_values(by=['Group', label+'_mean', label+'_median'], ascending=[True, False, False], inplace=True, axis=0)

        Xg_cor   = All_import.Group.unique()
        cor_dict = dict(zip(Xg_cor, color_[:len(Xg_cor)]))
        All_import['ColorsX'] = All_import.Group.map(cor_dict)

        All_raw = All_import[label]

        column  = sorted(set(All_raw.columns))
        X_labels = list( All_raw.index )

        Y_sd_min = All_import[label+'_mean'] - 1*All_import[label+'_std']
        Y_sd_max = All_import[label+'_mean'] + 1*All_import[label+'_std']
        #plt.plot(All_import['0_mean'], 'k-.', lw=1.5, label='mean_import', alpha=0.9)
        plt.figure(figsize=(13,10))
        plt.fill_between( X_labels, Y_sd_min, Y_sd_max,
                        color='grey', alpha=0.25, label=r'$\pm$ 1 std. dev.')

        legend_elements=[ Patch(facecolor=cor_dict[g], edgecolor='r', label=g) for g in sorted(cor_dict.keys()) ]
        legend_elements.append(Line2D([0], [0], color='r', linestyle='-.', lw=1.5, alpha= 0.9, label='%s mean'%label) )
        legend_elements.append(Line2D([0], [0], color='r', linestyle='-',  lw=1.5, alpha= 0.9, label='%s median'%label) )
        legend_elements.append(Patch(facecolor='grey', edgecolor='black' , alpha=0.3, label=r'$\pm$ 1 std. dev.') )
        if All_import[label+'_mean'].min() <0:
            legend_elements.append(Patch(facecolor='white',  edgecolor='blue',  label=r'%s $\geq$0'%label) )
            legend_elements.append(Patch(facecolor='white',  edgecolor='red' ,  label=r'%s <0'%label) )

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

        color_b = ['red' if i <  0 else 'blue' for i in All_import[label+'_mean']]
        linsy_b = ['-.'  if i <  0 else '--'   for i in All_import[label+'_mean']]
        color_c = list( All_import['ColorsX'] )
        for i, patch in enumerate(bplot['boxes']):
            patch.set(color=color_b[i], linewidth=1.3)
            patch.set(facecolor = color_c[i])

        for element in [ 'means','medians','fliers']:     #['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
            for i, patch in enumerate(bplot[element]):
                patch.set(color=color_b[i], linewidth=1)

        plt.title('The features %s of %s with %s estimator'%(label, _T_name, model))
        plt.legend(handles=legend_elements, ncol=ncol_, prop={'size':11}, loc='upper right')
        plt.ylabel( '%s of %s with %s estimator'%(label, _T_name, model))
        plt.xticks(rotation='270')
        plt.savefig(self.out, bbox_inches='tight')

class Evaluate(Baseset):
    def Dynamic_ROC(self, All_AUCs_, _T_name, model ):
        cv = len(All_AUCs_)
        color_ = self.color_*cv
        linestyle_ = self.linestyle_*cv
        markertyle_ = self.markertyle_*cv
        for i, [tau_terg, AUC_dynamic_te, AUC_mean_te] in enumerate(All_AUCs_):
            plt.plot(tau_terg, AUC_dynamic_te, marker=markertyle_[i], markersize=3.2, color=color_[i], linestyle=linestyle_[i], lw=1.0, label='mean AUC (%0.4f)'%AUC_mean_te )
            #plt.hlines(AUC_mean_te, tau_terg.min(), tau_terg.max(), color=color_[i], linestyle=linestyle_[i], lw=1.2 )

        plt.title('The %s dynamic AUCs of %s with %s estimator'%(cv, _T_name, model))
        plt.legend(loc='lower right')
        plt.ylabel('The %s time-dependent AUC'%cv)
        plt.xlabel('Times')
        plt.ylim([0.0, 1.0])
        plt.savefig( self.out )
        plt.close()

    def Dynamic_ROC_M(self, All_AUC_pd, _T_name, model ):
        Sco_mean = All_AUC_pd.mean(1, skipna=True)
        Sco_Std  = All_AUC_pd.std( 1, skipna=True)
        Sco_Ste  = All_AUC_pd.sem( 1, skipna=True) * 1.96

        plt.plot(All_AUC_pd.mean(1, skipna=True),   'r-', lw=1.5, label='mean_AUC'  ,alpha= 0.9 )
        plt.plot(All_AUC_pd.median(1, skipna=True), 'b--', lw=1.0, label='median_AUC',alpha= 0.6 )
        #plt.fill_between(All_AUC_pd.index, Sco_mean-1*Sco_Std, Sco_mean + 1*Sco_Std, color='grey', alpha=0.25, label=r'$\pm$ 1 std. dev.')
        plt.fill_between(All_AUC_pd.index, Sco_mean-1*Sco_Ste, Sco_mean + 1*Sco_Ste, color='grey', alpha=0.25, label=r'95% CI')

        plt.title('The dynamic AUCs of %s with %s estimator'%(_T_name, model))
        plt.legend(loc='lower right')
        plt.ylabel('The time-dependent AUC')
        plt.xlabel('Times')
        plt.ylim([0.0, 1.0])
        plt.savefig( self.out )
        plt.close()

    def Risk_Score(self, All_Score, All_risks,  _T_name, _S_name ):
        All_Score.index = [ 'S_' + str(i) for i in All_Score.index ]
        All_risks.index = [ 'S_' + str(i) for i in All_risks.index ]

        fsz = (17, 8) if All_Score.shape[0]<100 else (32, 15)
        fig = plt.figure(figsize = fsz ) #All_Score.shape[0]/4,All_Score.shape[0]/9) )
        fig.suptitle('The %s risk scores distribution'%(_T_name), x=0.5,y =0.90 )
        gs  = gridspec.GridSpec(24, 1)

        '''
        ax3 = plt.subplot(gs[0:1, :])
        ax3.pcolormesh(All_Score[[_S_name]].T, cmap=plt.cm.winter)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_ylabel(None)
        '''

        ax1 = plt.subplot(gs[0:9, :])
        ax1.grid(True)
        ax1.plot(All_Score[_T_name], 'c-' , marker='s', linewidth=1.5, markersize=4.2, label=_T_name)
        ax1.pcolor(All_Score[[_S_name]].T, cmap=plt.cm.winter)
        ax1.set_xticks([])
        ax1.set_ylabel(_T_name)
        ax1.legend()
        ax1.grid(which='both', axis='both')
        ax1.set_xlim(All_Score.index[0], All_Score.index[-1])

        ax2 = plt.subplot(gs[9:24, :] )
        ax2.plot(All_Score['risk_score_mean'], color=plt.cm.tab10(0), marker='*', linewidth=1.3, linestyle='--', markersize=4.2, label='risk_score_mean')
        ax2.plot(All_Score['risk_score_median'], color=plt.cm.tab10(1), marker='P', linewidth=1.3, linestyle=':', markersize=4.2, label='risk_score_median')
        ax2.plot(All_Score['risk_score_coefs_mean'], color=plt.cm.tab10(2) , marker='s', linewidth=1.3, linestyle='-.',markersize=4.2, label='risk_score_coefs_mean')
        ax2.scatter(All_risks.index, All_risks['risk_score'], color=plt.cm.tab10(3), marker='o', s=15.2, label='risk_score_all')
        ax2.set_xticklabels( All_Score.index, rotation='270')
        ax2.set_ylabel('risk_scores')
        ax2.set_xlim(All_Score.index[0], All_Score.index[-1])
        ax2.legend()
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

class Plots(Baseset):

    def Feature_SorceCV(self):
        (select_Sco, select_Fea, N, Y, cv) = self.array

        color_ = self.color_*select_Sco.shape[1]
        linestyle_ = self.linestyle_*select_Sco.shape[1]
        markertyle_ = self.markertyle_*select_Sco.shape[1]

        Sco_mean = select_Sco.mean(axis=1)
        Sco_Std  = select_Sco.std(axis=1)
        Fea_mean = select_Fea.mean()
        Fea_Std  = select_Fea.std()
        Fea_min  = (Fea_mean-1*Fea_Std) if (Fea_mean-1*Fea_Std) > 0 else 1
        Fea_max  = (Fea_mean + 1*Fea_Std) if (Fea_mean + 1*Fea_Std)<= select_Sco.shape[0] else select_Sco.shape[0]
        for i in range(select_Sco.shape[1]):
            plt.plot(select_Sco[i], marker=markertyle_[i], markersize=3.2, color=color_[i], linestyle=linestyle_[i], lw=1.0, label='')
            plt.vlines(select_Fea[i], 0, 1, color=color_[i], linestyle=linestyle_[i], lw=1.2, label='')
        plt.plot(Sco_mean , 'k-', lw=1.5, label='mean_score')
        plt.vlines(Fea_mean, 0, 1, lw=1.5, label='')
        plt.fill_between(Sco_mean.index, Sco_mean-1*Sco_Std, Sco_mean + 1*Sco_Std, color='grey', alpha=0.25, label=r'$\pm$ 1 std. dev.')
        plt.axvspan(Fea_min, Fea_max , alpha=0.3, color='grey', label='')

        plt.title('The %s features score of %s with %s estimator'%(cv, Y, N))
        plt.legend(loc='lower right')
        plt.ylabel('The %s features score'%cv)
        plt.xlabel('n featrues')
        plt.ylim([0.0, 1.0])
        plt.savefig('%sClass.Regress_%s_%s_%s_features_score.pdf' % (self.arg.outdir, Y, N ,cv))
        plt.close()

    def Feature_CoefSC(self):
        (select_Sco, select_Fea, N, Y, cv) = self.array

        Sco_mean = np.array(select_Sco).mean()
        Sco_Std  = np.array(select_Sco).std()
        Fea_mean = select_Fea.mean()
        Fea_Std  = select_Fea.std()

        Sco_min  = Sco_mean - Sco_Std
        Sco_max  = Sco_mean + Sco_Std

        Fea_min  = (Fea_mean-1*Fea_Std) if (Fea_mean-1*Fea_Std) > 0 else 1
        Fea_max  = (Fea_mean + 1*Fea_Std)

        plt.plot(select_Fea, select_Sco, 'g--', marker='*', linewidth=1.3, markersize=4.2, label='')
        plt.hlines(Sco_mean, 0, Fea_max+1, lw=1.5, label='mean_score')
        plt.vlines(Fea_mean, 0, 1, lw=1.5, label='mean_featrues')
        plt.axvspan(Fea_min, Fea_max , alpha=0.3, color='red', label='')
        plt.axhspan(Sco_min, Sco_max , alpha=0.3, color='blue', label='')

        plt.title('The %s features score of %s with %s estimator'%(cv, Y, N))
        plt.legend(loc='lower right')
        plt.ylabel('The %s features score'%cv)
        plt.xlabel('n featrues')
        plt.ylim([0.0, 1.0])
        plt.savefig('%sClass.Regress_%s_%s_%s_features_score.pdf' % (self.arg.outdir, Y, N ,cv))
        plt.close()

    def ROC_Predict_Import(self):
        (All_ROC, N, Y) = self.array

        color_ = self.color_*len(All_ROC)
        linestyle_ = self.linestyle_*len(All_ROC)
        markertyle_ = self.markertyle_*len(All_ROC)

        for i in range(len(All_ROC)):
            fpr, tpr, roc_auc, Y_test_accuracy, N = All_ROC[i]
            label_i = 'ROC curve (area = %0.2f), accuracy (%0.2f) %s' % (roc_auc, Y_test_accuracy, N)
            plt.plot(fpr, tpr, color=color_[i], lw=1.3, linestyle=linestyle_[i], label=label_i)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(Y + '_' + N + ' ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('%sClass_%s_%s_Predict_ROC_curve.pdf' % (self.arg.outdir, Y, N))
        plt.close()
