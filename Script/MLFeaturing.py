#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* Author : Zhou Wei                                       *
* Date   : Fri Jul 27 12:36:42 2018                       *
* E-mail : welljoea@gmail.com                             *
* You are using the program scripted by Zhou Wei.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''
from sklearn.feature_selection import (VarianceThreshold,  GenericUnivariateSelect,
                                       SelectKBest, SelectFromModel,
                                       f_classif, f_regression,
                                       chi2, RFECV,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             precision_recall_curve,
                             roc_curve, auc, r2_score,
                             average_precision_score)
#from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, StratifiedKFold

from sklearn.utils.validation import check_array
import joblib
import os
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from functools import partial
from itertools import combinations
from scipy.stats import (wilcoxon, ranksums, mannwhitneyu, ttest_ind,
                         pearsonr, spearmanr, kendalltau, linregress)

from .MLPlots import ClusT, MPlot
from .MLEstimators import ML
from .MLOpenWrite import OpenM, Openf
from .MLUnsupervising import Decomposition
from .MLNewAttribus import  CoxPHFitter_, StratifiedShuffleSplit_



class Featuring():
    def __init__(self, arg, log, *array, score=None, model='CoxPH', Type='S',  **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts
        if Type=='S':
            self.parameters, self.estimator= ML(self.model, SearchCV = self.arg.SearchCV).Regression()

    def UnivarCoxPH(self, Data, _X_names, _T_name, _S_name ):
        Coxcoef = []
        for _x in _X_names:
            cph = CoxPHFitter_()
            cph.fit(Data[[_x, _T_name, _S_name]],
                    duration_col=_T_name,
                    event_col=_S_name,
                    show_progress=False)

            SUMY = cph._summary().summary
            SUMY.loc[_x, 'CI_score'] = cph._summary().Concordance_
            Coxcoef.append(SUMY)
        Coxcoef = pd.concat(Coxcoef,axis=0)
        CoxcoefKEEP = Coxcoef[( (Coxcoef.p <=self.arg.UniP) & (Coxcoef.CI_score>=self.arg.CIs) )]
        self.log.CIF( ('%s: UnivarCoxPH'%_T_name).center(45, '-') )
        self.log.CIF( 'drop Feature: %s' % ( Coxcoef.shape[0] - CoxcoefKEEP.shape[0] ))
        self.log.CIF( 'keep Feature: %s'  % CoxcoefKEEP.shape[0] )
        self.log.NIF( 'Feature values:\n%s' % CoxcoefKEEP)
        self.log.CIF(45 * '-')
        return CoxcoefKEEP.index.to_list()

    def SelectKBest_R(self, Xdf, Ydf):
        def tranvisR(X, y, stat=''):
            f = []
            for n in range( X.shape[1]):
                s_p  = stat( X[:, n], y )
                if 'correlation' in dir(s_p):
                    f.append( [ s_p.correlation, s_p.pvalue ] )
                elif 'rvalue' in dir(s_p):
                    f.append( [ s_p.rvalue, s_p.pvalue ] )
                else:
                    f.append( [ s_p[0], s_p[1] ] )
            f = np.array(f)
            return( np.abs(f[:,0]), f[:,1] )

        k_F = int(self.arg.SelectK if ( self.arg.SelectK > 1) else round(self.arg.SelectK*len(Xdf.columns), 0))
        Fearture_select ={
            'VTh'  : VarianceThreshold(threshold=0.8*(1-0.8)),
            'ANVF' : SelectKBest(f_regression, k=k_F),
            'MI'   : SelectKBest(score_func=partial(mutual_info_regression, random_state=0), k=k_F),
            'PEAS' : SelectKBest(score_func=partial(tranvisR, stat=pearsonr  ), k=k_F),
            'SPM'  : SelectKBest(score_func=partial(tranvisR, stat=spearmanr ), k=k_F),
            'KDT'  : SelectKBest(score_func=partial(tranvisR, stat=kendalltau), k=k_F),
            'LR'   : SelectKBest(score_func=partial(tranvisR, stat=linregress), k=k_F),
            #'PRS'  : SelectKBest(lambda Xi, Yi: np.array(list(map(lambda x:pearsonr(x, Yi), Xi.T))).T[0], k=k_F),
        }

        Fearture_bool = []
        Fearture_name = []
        Fearture_PVS  = []
        for j in self.arg.SelectB:
            if j in Fearture_select.keys():
                i = Fearture_select[j]
                F_t = i.fit_transform(Xdf, Ydf)
                G_s = i.get_support()

                if j in [ 'MI' ]:
                    P_v = i.scores_
                elif j in [ 'VTh' ]:
                    P_v = i.variances_
                else:
                    P_v = i.pvalues_

                Fearture_name.append(j)
                Fearture_bool.append(G_s)
                Fearture_PVS.append(P_v)

        Fearture = pd.DataFrame(np.array(Fearture_bool).T, columns=Fearture_name,index=Xdf.columns)
        Fearture['True_ratio'] = Fearture.mean(1)

        FearturP = pd.DataFrame(np.array(Fearture_PVS ).T, columns=Fearture_name,index=Xdf.columns)
        Fearture_final = Fearture[ (Fearture['True_ratio'] >= self.arg.SelectR) ]
        Fearture_drop  = list( set(Fearture.index) - set( Fearture_final.index) )

        Openf('%s%s_Survival_SelectkBset.xls'%(self.arg.output, Ydf.name),Fearture).openv()
        Openf('%s%s_Survival_SelectkBset.Pvalues.xls'%(self.arg.output, Ydf.name),FearturP).openv()

        self.log.CIF( ('%s: SelectKBest_R'%Ydf.name).center(45, '-') )
        self.log.CIF( 'Raw   Feature: %s' % Fearture.shape[0]  )
        self.log.CIF( 'drop  Feature: %s\n%s' % ( len(Fearture_drop), Fearture_drop) )
        self.log.CIF( 'final Feature: %s'  % Fearture_final.index.to_list() )
        self.log.NIF( 'Feature bools:\n%s' % Fearture)
        self.log.CIF(45 * '-')
        return Fearture_final.index.to_list()

    def RFECV_S(self, Dfall, _Xa, _Y_T, _Y_S):
        X = Dfall[_Xa]
        Y = Dfall[[_Y_S + '_B', _Y_T ]].to_records(index = False)
        select_Sup = []
        select_Sco = []

        for i in range(self.arg.SelectCV_rep):   #pool
            for n in np.arange(0.15, 0.35, 0.01) :
                SSS =  StratifiedShuffleSplit_(n_splits=10, test_size=n, random_state=int(n*100))
                selector = RFECV(self.estimator,
                                    step=1,
                                    cv=SSS,
                                    n_jobs=self.arg.n_job,
                                    scoring =self.score)
                selector = selector.fit(X, Y)
                select_Sco.append(selector.grid_scores_)
                select_Sup.append(selector.ranking_)
                self.log.CIF('RFECV: %s -> %s , split %.2f '%(self.arg.model, self.model, n))

        select_Sco = pd.DataFrame(np.array(select_Sco).T, index=range(1, X.shape[1]+1))
        select_Sup = pd.DataFrame(np.array(select_Sup).T, index=X.columns)
        select_Sco.columns = range(select_Sco.shape[1])
        select_Sup.columns = [ '%s_%s'%(self.model, i) for i in range(select_Sup.shape[1]) ]
        select_feature = (select_Sup==1).sum(0).values

        Openf('%s%s_Survival_RFECV_%s_ranking.xls'%(self.arg.output, _Y_T, self.model),select_Sup).openv()
        MPlot('%s%s_Survival_RFECV_%s_Fscore.pdf'%(self.arg.output, _Y_T, self.model)).Feature_Sorce(
            select_Sco, select_feature, self.model, _Y_T, 'RFECV' )

        return( select_Sup )

    def SFSCV_S(self, Dfall, _Xa, _Y_T, _Y_S):
        X = Dfall[_Xa]
        Y = Dfall[[_Y_S + '_B', _Y_T ]].to_records(index = False)

        select_Sup = []
        select_Sco = []
        k_features = ()

        if set(self.arg.k_features) & set(['best', 'parsimonious']):
            k_features = self.arg.k_features[0]
        elif len(self.arg.k_features) == 1:
            k_features = int(float(self.arg.k_features[0]) * len(_Xa))
        elif len(self.arg.k_features) == 2:
            k_features = tuple( [ int(float(i) * len(_Xa)) for i in self.arg.k_features ] )

        for i in range(self.arg.SelectCV_rep):   #pool
            for n in np.arange(0.15, 0.35, 0.01) :
                SSS =  StratifiedShuffleSplit_(n_splits=10, test_size=n, random_state=int(n*100))
                selector = SFS(self.estimator,
                        k_features= k_features, 
                        forward=True, 
                        floating=False, 
                        verbose=0,
                        scoring=self.score,
                        n_jobs=self.arg.n_job,
                        cv=SSS)
                selector = selector.fit(X, Y)
                select_feat = selector.k_feature_names_
                select_feat = pd.DataFrame([[1]]*len(select_feat),index=select_feat)
                avg_score = pd.DataFrame(selector.subsets_).loc['avg_score']
                select_Sup.append(select_feat)
                select_Sco.append(avg_score)
                self.log.CIF('SFSCV: %s -> %s , split %.2f '%(self.arg.model, self.model, n))

        select_Sup = pd.concat(select_Sup,axis=1,sort=False).fillna(0).astype(int)
        select_Sco = pd.concat(select_Sco,axis=1,sort=False).fillna(0)
        select_Sco.columns = range(select_Sco.shape[1])
        select_Sup.columns = [ '%s_%s'%(self.model, i) for i in range(select_Sup.shape[1]) ]
        select_feature = (select_Sup==1).sum(0).values

        Openf('%s%s_Survival_SFSCV_%s_ranking.xls'%(self.arg.output, _Y_T, self.model),select_Sup).openv()
        MPlot('%s%s_Survival_SFSCV_%s_Fscore.pdf'%(self.arg.output,  _Y_T, self.model)).Feature_Sorce(
            select_Sco, select_feature.values, self.model, _Y_T, 'SFSCV' )

        return( select_Sup )

class Feature_selection():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.Type  = 'S'
        self.arg.output = '%s/02FeatureSLT/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs( os.path.dirname(self.arg.output), exist_ok=True)

    def Serial(self, _All_FS_models, _XYdf, _X_names, _T_name, _S_name ):
        _Xdf = _XYdf[_X_names]
        for _modeli, _packai, _rate_i in _All_FS_models:
            _set_Rank = []
            for model_j in _modeli:
                self.log.CIF('the %s model use the %s estimator for %s.'%(self.arg.model, model_j, _packai ))
                _i_Rank = eval( 'Featuring(self.arg, self.log, model =model_j, Type=self.Type ).%s(_XYdf, _X_names, _T_name, _S_name)'%_packai ) 
                _set_Rank.append(_i_Rank)

            _set_Rank = pd.concat(_set_Rank, axis=1)
            _set_Rank['SUM_Rank'] = _set_Rank.sum(1)
            _set_Rank['SUM_Bool'] = (_set_Rank==1).sum(1)
            _set_Rank.sort_values(by=['SUM_Bool', 'SUM_Rank'], ascending=[False, True], inplace = True)

            if 0 < _rate_i <=1 :
                _rate_i  =  int(_set_Rank.shape[1] * _rate_i)
            if (_rate_i > _set_Rank.shape[0]) or (_rate_i == 0 ):
                _rate_i = _set_Rank.shape[0]

            B_number = _set_Rank.iloc[int(_rate_i-1), : ]['SUM_Bool']
            if B_number ==0: B_number += 1
            _set_Rank = _set_Rank[_set_Rank['SUM_Bool'] >= B_number]

            Drop = list( set( _Xdf.columns) - set(_set_Rank.index))
            KEEP = _set_Rank.index.tolist()
            _Xdf = _Xdf[KEEP]

            self.log.CIF( ('%s: SelectUseModel'%_T_name).center(45, '-') )
            self.log.CIF( '%s drop %s Features: %s'%( _packai, len(Drop), Drop ) )
            self.log.CIF( '%s keep %s Features: %s'%( _packai, len(KEEP), KEEP ) )
            self.log.NIF( 'Features Selected: \n%s' % _set_Rank )
            self.log.CIF(45 * '-')

        return(KEEP)

    def Parallel(self, _All_FS_models, _XYdf, _X_names, _T_name, _S_name ):
        _Xdf = _XYdf[_X_names]
        _set_Rank = []
        _set_pack = []
        _set_rate = []
        for _modeli, _packai, _rate_i in _All_FS_models:
            _set_pack.append(_packai)
            _set_rate.append(_rate_i)
            for model_j in _modeli:
                self.log.CIF('the %s model use the %s estimator for %s.'%(self.arg.model, model_j, _packai ))
                _i_Rank = eval( 'Featuring(self.arg, self.log, model =model_j, Type=self.Type ).%s(_XYdf, _X_names, _T_name, _S_name)'%_packai ) 
                _set_Rank.append(_i_Rank)

        _set_Rank = pd.concat(_set_Rank, axis=1, sort =False)
        _set_Rank['SUM_Rank'] = _set_Rank.sum(1)
        _set_Rank['SUM_Bool'] = (_set_Rank==1).sum(1)
        _set_Rank.sort_values(by=['SUM_Bool', 'SUM_Rank'], ascending=[False, True], inplace = True)

        _rate_m = _set_rate[0]
        _pack_a = '+'.join(_set_pack)

        if 0 < _rate_m <=1 :
            _rate_m  =  int(_set_Rank.shape[1] * _rate_m)
        if (_rate_m > _set_Rank.shape[0]) or (_rate_m == 0 ):
            _rate_m = _set_Rank.shape[0]

        B_number = _set_Rank.iloc[int(_rate_m-1), : ]['SUM_Bool']
        if B_number ==0: B_number += 1
        _set_Rank = _set_Rank[_set_Rank['SUM_Bool'] >= B_number]

        Drop = list( set( _Xdf.columns) - set(_set_Rank.index))
        KEEP = _set_Rank.index.tolist()
        _Xdf = _Xdf[KEEP]

        self.log.CIF( ('%s: SelectUseModel'%_T_name).center(45, '-') )
        self.log.CIF( '%s drop %s Features: %s'%( _pack_a, len(Drop), Drop ) )
        self.log.CIF( '%s keep %s Features: %s'%( _pack_a, len(KEEP), KEEP ) )
        self.log.NIF( 'Features Selected: \n%s' % _set_Rank )
        self.log.CIF(45 * '-')

        return(KEEP)

    def SelectUseModel(self, _XYdf, _X_names, _T_name, _S_name):
        if self.arg.specifM:
            RFE_model = self.arg.specifM
            SFS_model = self.arg.specifM
        else:
            SFS_model = [self.arg.model]
            RFE_model = [self.arg.model]

        _All_FS_models = []
        if self.arg.RFECV:
            _All_FS_models.append([RFE_model, 'RFECV_S', self.arg.RFE_rate])
        if self.arg.SFSCV:
            _All_FS_models.append([SFS_model, 'SFSCV_S', self.arg.SFS_rate])

        if len(_All_FS_models) > 0:
            if self.arg.set   == 'parallel':
                return(self.Parallel(_All_FS_models, _XYdf, _X_names, _T_name, _S_name) )
            elif self.arg.set == 'serial':
                return(self.Serial(_All_FS_models, _XYdf, _X_names, _T_name, _S_name) )

    def Fselect(self):
        (group, AYa, RYa, CYa, SYa, Xa, Xg) = OpenM(self.arg, self.log).opens()
        Standfile =  '%s/01PreProcess/%sTrainTest.standard.data.xls'%(self.arg.outdir, self.arg.header )
        dfall = Openf(Standfile, index_col=0).openb()
        _XALL = [i for i in dfall.columns if i in Xa]

        for sgroup, sterm in SYa.iterrows():
            self.Type = 'S'
            _S_name = sterm['S']
            _T_name = sterm['T']
            dfall[_S_name +'_B'] = dfall[_S_name].apply(lambda x: True if x >0 else False)

            _Xdf = dfall[_XALL]
            _Y_S = dfall[_S_name]
            _Y_T = dfall[_T_name]

            KEEP = _XALL
            self.log.CIF( ('%s: Feature Selecting'% _T_name ).center(45, '*') )

            if self.arg.SelectB:
                KEEP = Featuring(self.arg, self.log).SelectKBest_R( _Xdf, _Y_T)
                KEEP = Featuring(self.arg, self.log).UnivarCoxPH(dfall, KEEP, _T_name, _S_name)
            KEEP = self.SelectUseModel(dfall, KEEP, _T_name, _S_name)

            Final_XYdf = dfall[ KEEP + [_S_name, _T_name] ]
            Openf('%s%s_Survival_FeatureSLT.Data.xls'%(self.arg.output, _T_name), Final_XYdf).openv()
            ClusT('%s%s_Survival_FeatureSLT.Data_complete.pdf'%(self.arg.output, _T_name) ).Plot_heat(dfall[KEEP] , dfall[[_S_name, _T_name]], Xg, method='complete')
            ClusT('%s%s_Survival_FeatureSLT.Data_average.pdf' %(self.arg.output, _T_name) ).Plot_heat(dfall[KEEP] , dfall[[_S_name, _T_name]], Xg, method='average' )

            Decomposition(self.arg, self.log).PCA( Final_XYdf, KEEP, [_S_name, _T_name], '%s%s_Survival_FeatureSLT'%(self.arg.output, _T_name) )

            self.log.CIF( ('%s: Feature Selecting Finish'%_T_name).center(45, '*') )


