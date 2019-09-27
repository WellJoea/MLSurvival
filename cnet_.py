
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

def NET_(net):
    cphe, cphp = net
    cphe.fit(DF[_Xa], Y_df)
    #print(cphe.coef_.shape, cphe.alphas_,1111) # np.exp(cphe.coef_), cphe.score(DF[_Xa], Y_df))
    #print(cphe.coef_[:, -1], 2222)
    #print(cphe.predict(DF[_Xa]),999999999999999999)
    #print(cphe.coef_)
    #plt.plot(cphe.alphas_, safe_sqr(cphe.coef_).T)
    #plt.plot(cphe.alphas_,cphe.deviance_ratio_)
    alp1 = cphe.alphas_
    coe1 = cphe.coef_.T 
    pre1 = cphe.predict(DF[_Xa])
    sco1 = cphe.score(DF[_Xa], Y_df)

    '''
    from copy import copy
    es = copy( cphe )
    ps = {"alphas": [[v] for v in alp1[-10:]]}
    glf = GridSearchCV( es , ps ,
                    n_jobs=-1, 
                    cv=10,
                    scoring=None,
                    error_score = np.nan,
                    return_train_score=True,
                    refit = True,
                    iid=True)
    glf.fit(DF[_Xa], Y_df)

    print('alpx', glf.best_estimator_.alphas_)
    print('coex', glf.best_estimator_.coef_)
    print('scox', glf.score(DF[_Xa], Y_df))
    print('pare', glf.best_params_)
    '''

    scorcs= []
    ders = []
    for i in alp1:
        cphe.set_params(alphas= [i])
        cphe.fit(DF[_Xa], Y_df)
        scorcs.append(cphe.score(DF[_Xa], Y_df))
        ders.append(cphe.deviance_ratio_)
    scorcs = np.array(scorcs)
    ders = np.array(ders)
    print( 'scorcs maxid', np.where(scorcs==scorcs.max()) )
    print( 'ders   maxid', np.where(ders  ==ders.max()) )
    aa_best= alp1[np.where(scorcs==scorcs.max())]
    cphe.set_params(alphas= aa_best)
    cphe.fit(DF[_Xa], Y_df)

    alp2 = cphe.alphas_
    coe2 = cphe.coef_.T 
    pre2 = cphe.predict(DF[_Xa])
    sco2 = cphe.score(DF[_Xa], Y_df)

    print('alp2', alp2)
    #print('max_id',ders )
    print('coe2', coe2)
    print('pre2', pre2)
    print('sco2', sco2)
    return (alp1, coe1 )
cbs_, ccf_ = NET_(NET())

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
        print()
        return self

    '''
    def _get_coef(self, alpha):
        check_is_fitted(self, "coef_")

        if alpha is None:
            coef = self.coef_[:, -1]
        else:
            coef = self._interpolate_coefficients(alpha)
        return coef

    def predict(self, X):
        super(CoxnetSurvivalAnalysis_, self).predict(X, alpha=None) 
    '''
def SNet_():
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
    alp3 = cphe.alphas_
    coe3 = cphe.coef_ 
    pre3 = cphe.predict(DF[_Xa])
    sco3 = cphe.score(DF[_Xa], Y_df)

    print('alp3', alp3)
    print('max_id',cphe.max_id )
    print('alphab_',cphe.alphab_)
    print('coe3', coe3)
    print('pre3', pre3)
    print('sco3', sco3)

    return (cphe.alphas_, cphe.coefs_.T )

als_ , cs_ = SNet_()
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10), gridspec_kw={'wspace': 0.1} ) #, sharex=True)
ax1.plot(cbs_, ccf_ )
ax2.plot(als_ , cs_ )
plt.show()
'''