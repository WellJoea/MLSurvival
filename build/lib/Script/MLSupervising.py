from sklearn.model_selection import (StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, 
                                     RepeatedKFold, RepeatedStratifiedKFold, train_test_split,
                                     GridSearchCV, RandomizedSearchCV )

from sklearn.utils._joblib import Parallel, delayed
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc

import joblib
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from .MLOpenWrite import OpenM, Openf
from .MLEstimators import ML
from .MLPlots import ClusT, Evaluate, MPlot

class Processing():
    def __init__(self, arg, log, *array, score=None, model='CoxPH', Type='S', **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts
        self.Type  = Type
        self.SSS   = StratifiedShuffleSplit(n_splits=10, test_size=0.3 , random_state=1)
        self.RKF   = RepeatedKFold(n_splits=4, n_repeats=3, random_state= 1)
        if Type=='S':
            self.parameters, self.estimator= ML(self.model, SearchCV = self.arg.SearchCV).Regression()

    def CrossvalidationSplit(self, Xdf, Ydf, Type='SSA', random_state = None ):
        if Type == 'SSS':
            CVS = StratifiedShuffleSplit(n_splits =self.arg.crossV,
                                         test_size=self.arg.testS,
                                         random_state=random_state).split(Xdf,Ydf)
        elif Type == 'SKF':
            CVS = StratifiedKFold(n_splits=self.arg.crossV,
                                  random_state=random_state).split(Xdf,Ydf)
        elif Type == 'RSKF':
            CVS = RepeatedStratifiedKFold(n_splits = 4,
                                          n_repeats= 3,
                                          random_state=random_state).split(Xdf,Ydf)
        elif Type == 'LOU':
            CVS = LeaveOneOut().split(Xdf)
        elif Type == 'SSA':
            CVS = StratifiedShuffleSplit(n_splits =self.arg.crossV,
                                         test_size=self.arg.testS,
                                         random_state=random_state).split(Xdf,Ydf)
        elif Type == 'RKF':
            CVS = RepeatedKFold(n_splits = 4,
                                n_repeats= 3,
                                random_state=random_state).split(Xdf)
        all_test = []
        all_split= []
        for train_index, test_index in CVS:
            all_test.extend(test_index)
            all_split.append( [train_index.tolist(), test_index.tolist()])

        if Type == 'SSA':
            if len(set(all_test))< len(Ydf):
                SFK = StratifiedKFold(n_splits=round(1/self.arg.testS) ,
                                      random_state=random_state).split(Xdf,Ydf)
                all_add = [ [tr.tolist(), te.tolist()] for tr, te in SFK]
                all_split += all_add

        return(all_split)

    def Clf_S(self, X_train, X_test, y_train, y_test):
        y_ttall =  np.concatenate( (y_train,y_test )) 
        name_event, name_time = y_train.dtype.names
        if self.arg.SearchCV == 'GSCV':
            clf = GridSearchCV(self.estimator, self.parameters,
                               n_jobs=self.arg.n_job,
                               cv=self.RKF,
                               scoring=self.score,
                               error_score = np.nan,
                               return_train_score=True,
                               refit = True,
                               iid=True)
        elif self.arg.SearchCV == 'RSCV':
            clf = RandomizedSearchCV(self.estimator, self.parameters,
                                     n_jobs=self.arg.n_job,
                                     cv=self.RKF,
                                     n_iter = self.arg.n_iter,
                                     scoring=self.score,
                                     return_train_score=True,
                                     iid=True,
                                     refit = True,
                                     error_score='raise')

        clf.fit(X_train, y_train)
        if hasattr(clf.best_estimator_, 'coef_'):
            coef_ = clf.best_estimator_.coef_
        elif hasattr(clf.best_estimator_, 'feature_importances_'):
            coef_ = clf.best_estimator_.feature_importances_
        else:
            raise RuntimeError('The classifier does not expose '
                               '"coef_" or "feature_importances_" '
                               'attributes')
        if coef_.ndim == 1:
            coef_ = coef_.reshape( coef_.shape[0], 1 )
        if coef_.shape[0]==1:
            coef_ = coef_.T
        coefs_ = pd.DataFrame( np.hstack((coef_, np.exp(coef_))), 
                               index=X_train.columns,
                               columns=['coef_','exp_coef_'],
                             )
        coefs_.sort_values(by=list(coefs_.columns), ascending=[False]*coefs_.shape[1], inplace =True)

        risk_tr  = clf.predict(X_train)
        risk_te  = clf.predict(X_test)
        risks_te = pd.DataFrame( np.vstack(( y_test[name_time], y_test[name_event], risk_te )).T, 
                                 index=X_test.index,
                                 columns=[name_time, name_event[:-2], 'risk_score'],
                                )
        risks_te.sort_values(by=list(risks_te.columns), ascending=[False]*risks_te.shape[1], inplace =True)

        tau_trrg = np.unique( np.percentile(y_train[name_time], np.linspace(1, 99, 99)) )
        tau_terg = np.unique( np.percentile(y_test[name_time] , np.linspace(5, 95, 99)) )

        CI_H_tr  = clf.score(X_train, y_train)
        CI_H_te  = clf.score(X_test, y_test)

        CI_U_tr  = concordance_index_ipcw(y_train, y_train, risk_tr, tau=None)[0]
        try:
            CI_U_te = concordance_index_ipcw(y_train, y_test , risk_te, tau=None)[0]
        except ValueError:
            self.log.CWA("censoring survival function is zero at one or more time points in CI_U")
            CI_U_te = concordance_index_ipcw(y_ttall, y_test , risk_te, tau=tau_terg[-1])[0]

        AUC_dynamic_tr, AUC_mean_tr = cumulative_dynamic_auc(y_train, y_train, risk_tr, tau_trrg)
        try:
            AUC_dynamic_te, AUC_mean_te = cumulative_dynamic_auc(y_train, y_test , risk_te, tau_terg)
        except ValueError:
            self.log.CWA("censoring survival function is zero at one or more time points in AUC")
            AUC_dynamic_te, AUC_mean_te = cumulative_dynamic_auc(y_ttall, y_test , risk_te, tau_terg)

        Best_Model = { 'Estimate' : clf.best_estimator_,
                       'Features' : X_train.columns.tolist(),
                       'y_ttall'  : y_ttall,
                     }
        self.log.CIF(('%s %s Model Fitting and hyperparametering'% ( name_time, self.model)).center(45, '-') )
        self.log.CIF( '%s best parameters: \n%s' %(self.model, clf.best_params_) )
        self.log.CIF( '%s best score: %s' %(self.model, clf.best_score_) )
        self.log.CIF( '%s best coefs: \n%s' %(self.model, coefs_) )
        #self.log.CIF( '%s train dynami AUC:\n%s'%(self.model, AUC_dynamic_tr) )
        self.log.CIF( '%s train mean   AUC: %s' %(self.model, AUC_mean_tr) )
        self.log.CIF( '%s train CI_H score: %s' %(self.model, CI_H_tr) )
        self.log.CIF( '%s train CI_U score: %s' %(self.model, CI_U_tr) )
        #self.log.CIF( '%s test  dynami AUC:\n%s'%(self.model, AUC_dynamic_te) )
        self.log.CIF( '%s test  mean   AUC: %s' %(self.model, AUC_mean_te) )
        self.log.CIF( '%s test  CI_H score: %s' %(self.model, CI_H_te) )
        self.log.CIF( '%s test  CI_U score: %s' %(self.model, CI_U_te) )
        self.log.CIF( 45 * '-')
        return (Best_Model, coefs_, risks_te, [tau_terg, AUC_dynamic_te, AUC_mean_te] )

    def ClfP_S(self, Predictioin, _pXa, _T_name, _S_name ):
        _Predict = Predictioin.copy()
        Best_Model = joblib.load('%s/03ModelFit/%s%s_Survival_best_estimator.pkl'%(self.arg.outdir, self.arg.header, _T_name))
        All_coefs_, All_risks_, All_AUCs_ = [], [], []
        for i, _model_i in enumerate(Best_Model):
            _X_Features = _model_i['Features']
            _clf        = _model_i['Estimate']
            y_ttall     = _model_i['y_ttall']
            X_pred      = _Predict[_X_Features]

            if hasattr(_clf, 'coef_'):
                coef_ = _clf.coef_
            elif hasattr(_clf, 'feature_importances_'):
                coef_ = _clf.feature_importances_
            else:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')
            if coef_.ndim == 1:
                coef_ = coef_.reshape( coef_.shape[0], 1 )
            if coef_.shape[0]==1:
                coef_ = coef_.T
            coefs_ = pd.DataFrame( np.hstack((coef_, np.exp(coef_))),
                                   index=_X_Features,
                                   columns=['coef_','exp_coef_'],
                                 )
            coefs_.sort_values(by=list(coefs_.columns), ascending=[False]*coefs_.shape[1], inplace =True)

            risk_pr  = _clf.predict(X_pred)
            risks_pr = pd.DataFrame( np.vstack(( _Predict[_T_name], _Predict[_S_name], risk_pr )).T, 
                                     index=_Predict.index, 
                                     columns=[_T_name, _S_name, 'risk_score'],
                                    )
            risks_pr.sort_values(by=list(risks_pr.columns), ascending=[False]*risks_pr.shape[1], inplace =True)

            '''
            blsurv   = _clf._baseline_model.baseline_survival_
            cumblh   = _clf._baseline_model.cum_baseline_hazard_
            cumhf_tr = _clf.predict_cumulative_hazard_function(X_pred)
            survf_tr = _clf.predict_survival_function(X_pred)
            '''

            if not _Predict[_T_name].isnull().any() :
                _Predict[_S_name +'_B'] = _Predict[_S_name].apply(lambda x: True if x >0 else False)
                Y_pred = _Predict[[_S_name + '_B', _T_name ]].to_records(index = False)

                tau_prrg = np.unique( np.percentile(_Predict[_T_name] , np.linspace(5, 95, 99)) )
                CI_H_pr  = _clf.score(X_pred, Y_pred)
                try:
                    CI_U_pr = concordance_index_ipcw(y_ttall, Y_pred , risk_pr, tau=None)[0]
                except ValueError:
                    self.log.CWA("censoring survival function is zero at one or more time points in CI_U")
                    CI_U_pr = concordance_index_ipcw(y_ttall, Y_pred , risk_pr, tau=tau_prrg[-1])[0]
                try:
                    AUC_dynamic_pr, AUC_mean_pr = cumulative_dynamic_auc(y_ttall, Y_pred , risk_pr, tau_prrg)
                except ValueError:
                    self.log.CWA("censoring survival function is zero at one or more time points in AUC")
                    y_ttpall = np.concatenate( (y_ttall, Y_pred ))
                    AUC_dynamic_pr, AUC_mean_pr = cumulative_dynamic_auc(y_ttpall, Y_pred , risk_pr, tau_prrg)

                All_AUCs_.append( [tau_prrg, AUC_dynamic_pr, AUC_mean_pr] )
            All_coefs_.append(coefs_)
            All_risks_.append(risks_pr)

            self.log.CIF(('%s %s Model Fitting and hyperparametering'% (_T_name, self.model)).center(45, '-') )
            self.log.CIF( '%s best parameters: \n%s' %(self.model, _clf) )
            self.log.CIF( '%s best coefs: \n%s' %(self.model, coefs_) )
            self.log.CIF( '%s predict  mean   AUC: %s' %(self.model, AUC_mean_pr) )
            self.log.CIF( '%s predict  CI_H score: %s' %(self.model, CI_H_pr) )
            self.log.CIF( '%s predict  CI_U score: %s' %(self.model, CI_U_pr) )
            self.log.CIF( 45 * '-')

        return ( All_coefs_, All_risks_, All_AUCs_  )

    def Coefs_S(self, All_coefs_, _T_name, Xg):
        All_coefs_ = pd.concat(All_coefs_, axis=1, sort=False).fillna(0)
        column_u   = [ 'coef_', 'exp_coef_' ]
        All_coefs_ = All_coefs_[column_u]
        for i in column_u:
            All_coefs_['%s_mean'%i]  = All_coefs_[i].mean(axis=1)
            All_coefs_['%s_std'%i]   = All_coefs_[i].std(axis=1)
            All_coefs_['%s_median'%i]= All_coefs_[i].median(axis=1)
            MPlot('%s%s_Survival_%s.boxI.pdf'  %(self.arg.output, _T_name, i)).Feature_Coefs_box( All_coefs_, Xg, i, _T_name, self.arg.model, sort_by_group=False)
            MPlot('%s%s_Survival_%s.boxII.pdf' %(self.arg.output, _T_name, i)).Feature_Coefs_box( All_coefs_, Xg, i, _T_name, self.arg.model, sort_by_group=True)

        All_coefs_['exp(coef_mean)'] = np.exp( All_coefs_['coef__mean'] )
        All_coefs_.sort_values(by=['%s_mean'%i for i in column_u], ascending=[False]*len(column_u), inplace=True, axis=0)

        Openf('%s%s_Survival_coefficients.xls' %(self.arg.output, _T_name), (All_coefs_)).openv()
        MPlot('%s%s_Survival_coefficients.lines.pdf' %(self.arg.output, _T_name)).Feature_Coefs( All_coefs_, _T_name, self.arg.model )

    def Score_S(self, All_risks_, All_coefs_, All_X, _T_name, _S_name,  header='TrainTest'):
        _coefs_  = pd.concat(All_coefs_, axis=1, sort=False).fillna(0)['coef_'].mean(axis=1)
        Mean_score = pd.DataFrame( np.dot( All_X[_coefs_.index], _coefs_), 
                                   index=All_X.index,
                                   columns=['risk_score_coefs_mean']
                                 )
        All_risks = pd.concat(All_risks_,axis=0)

        All_mean   = All_risks.groupby([All_risks.index]).mean()
        All_median = All_risks.groupby([All_risks.index]).median()
        All_Score  = All_mean[[_T_name, _S_name]]
        All_Score['risk_score_mean']   = All_mean['risk_score']
        All_Score['risk_score_median'] = All_median['risk_score']
        All_Score['risk_score_coefs_mean'] = Mean_score
        All_Score['exp_risk_score_mean']   = np.exp(All_mean['risk_score'])
        All_Score['exp_risk_score_median'] = np.exp(All_median['risk_score'])
        All_Score['exp_risk_score_coefs_mean'] = np.exp(Mean_score)
        All_Score.sort_values(by=list(All_Score.columns), ascending=[True]*All_Score.shape[1], inplace=True, axis=0)

        Openf('%s%s_Survival_%s_risk_score.detail.xls' %(self.arg.output, _T_name, header), (All_risks)).openv()
        Openf('%s%s_Survival_%s_risk_score.final.xls'  %(self.arg.output, _T_name, header), (All_Score)).openv()
        Evaluate('%s%s_Survival_%s_risk_score.pdf' %(self.arg.output, _T_name, header)).Risk_Score( All_Score, All_risks,  _T_name, _S_name)

    def DyROC_S(self, All_AUCs_, _T_name, tag='TrainTest'):
        All_AUC_pd = [ pd.Series(auc, index=time) for time, auc, _ in All_AUCs_ ]
        All_AUC_pd = pd.concat(All_AUC_pd, axis=1).fillna(np.nan)

        Openf('%s%s_Survival_%s_CV_dynamic_AUC.xls' %(self.arg.output, _T_name, tag,), (All_AUC_pd)).openv()
        Evaluate('%s%s_Survival_%s_CV_dynamic_AUC.pdf' %(self.arg.output, _T_name, tag)).Dynamic_ROC( All_AUCs_, _T_name, self.arg.model )
        Evaluate('%s%s_Survival_%s_CV_dynamic_AUC_mean.pdf' %(self.arg.output, _T_name, tag)).Dynamic_ROC_M( All_AUC_pd, _T_name, self.arg.model )

class Modeling():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.Type  = 'S'
        self.arg.output = '%s/03ModelFit/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs( os.path.dirname(self.arg.output), exist_ok=True)

    def Model_S(self, DFall, _X_names, _T_name, _S_name, Xg):
        All_estmt_, All_coefs_, All_risks_, All_AUCs_ = [], [], [], []
        CVM = Processing(self.arg, self.log).CrossvalidationSplit( DFall[_X_names] , DFall[_S_name], Type= self.arg.CVfit)
        for train_index, test_index in CVM:
            _Train , _Test   = DFall.iloc[train_index, :], DFall.iloc[test_index, :]
            X_train, y_train = _Train[_X_names], _Train[[_S_name + '_B', _T_name ]].to_records(index = False)
            X_test , y_test  = _Test[ _X_names], _Test[ [_S_name + '_B', _T_name ]].to_records(index = False)

            estimts_, coefs_, risks_te, AUCs_te = \
                Processing( self.arg, self.log, model=self.arg.model, Type = self.Type ).Clf_S( X_train, X_test, y_train, y_test )
            All_estmt_.append(estimts_)
            All_coefs_.append(coefs_)
            All_risks_.append(risks_te)
            All_AUCs_.append(AUCs_te)

        ### All_parameter
        joblib.dump(All_estmt_, '%s%s_Survival_best_estimator.pkl' %(self.arg.output, _T_name), compress=1)
        ### All_coefs_
        Processing( self.arg, self.log, model=self.arg.model).Coefs_S(All_coefs_, _T_name, Xg)
        ### All_risks_
        Processing( self.arg, self.log, model=self.arg.model).Score_S(All_risks_, All_coefs_, DFall[_X_names], _T_name, _S_name)
        ### All_AUCs_
        Processing( self.arg, self.log, model=self.arg.model).DyROC_S(All_AUCs_, _T_name, tag='TrainTest')

    def Fitting(self):
        (group, AYa, RYa, CYa, SYa, Xa, Xg) = OpenM(self.arg, self.log).opens()
        for sgroup, sterm in SYa.iterrows():
            self.Type  = 'S'
            _S_name = sterm['S']
            _T_name = sterm['T']

            if self.arg.pca:
                Fsltfile = '%s/02FeatureSLT/%s%s_Survival_FeatureSLT.PCA.xls' %(self.arg.outdir, self.arg.header, _T_name )
            else :
                Fsltfile = '%s/02FeatureSLT/%s%s_Survival_FeatureSLT.Data.xls'%(self.arg.outdir, self.arg.header, _T_name )

            DFall = Openf(Fsltfile, index_col=0).openb()
            _X_names = list( set(DFall.columns)- set([_S_name, _T_name]) )
            DFall[_S_name +'_B'] = DFall[_S_name].apply(lambda x: True if x >0 else False)

            self.log.CIF( ('%s: Supervised MODELing'%_T_name).center(45, '*') )
            self.log.NIF( '%s Status Counts:\n%s' %(_T_name, DFall[_S_name].value_counts().to_string()) )
            self.Model_S( DFall, _X_names, _T_name, _S_name, Xg )

            self.log.CIF( ('%s: Supervised MODELing Finish'%_T_name).center(45, '*') )

