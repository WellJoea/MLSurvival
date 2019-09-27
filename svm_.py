
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import Ridge, Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
#import scikitplot as skplt
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib

import platform
if platform.system()=='Linux':
    #matplotlib.use('Agg')
    pass
elif platform.system()=='Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')
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

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sklearn.model_selection import (StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, 
                                     RepeatedKFold, RepeatedStratifiedKFold, train_test_split,
                                     GridSearchCV, RandomizedSearchCV )
                                
from sklearn.utils import safe_sqr
from sksurv.svm import (FastKernelSurvivalSVM, FastSurvivalSVM, MinlipSurvivalAnalysis, 
                        HingeLossSurvivalSVM, NaiveSurvivalSVM)

from sklearn.feature_selection import RFECV

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


X_train = DF[_Xa].iloc[:121,]
X_test  = DF[_Xa].iloc[121:,] 

Y_train = DF[[_S_Time+'_B', _T_Time ]].iloc[:121,:].to_records(index = False)
Y_test  = DF[[_S_Time+'_B', _T_Time ]].iloc[121:,:].to_records(index = False) 


FKSVM = FastKernelSurvivalSVM(alpha=1,
                              rank_ratio=1,
                              fit_intercept=False,
                              kernel='rbf',
                              gamma=None,
                              degree=3,
                              coef0=1,
                              kernel_params=None,
                              max_iter=20,
                              verbose=False,
                              tol=None,
                              optimizer=None,
                              random_state=None,
                              timeit=False,
                              )
FKSVMP = {'alpha'     : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3],
          'optimizer' : ["avltree", "rbtree"],
          'rank_ratio': np.arange(0, 1, 0.005),
          'max_iter'  : [10, 20, 30],
          'kernel'    : ["linear",  "poly", "rbf", "sigmoid", "cosine", "precomputed"], 
          'degree'    : [2,3,4,5], 
          'timeit'    : [False],
          }

FSVM = FastSurvivalSVM( alpha=1,
                        rank_ratio=1,
                        fit_intercept=False,
                        max_iter=20,
                        verbose=False,
                        tol=None,
                        optimizer='PRSVM',
                        random_state=None,
                        timeit=False,
                        )

FSVMP = [{'alpha'     : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3],
          'optimizer' : ["avltree", "direct-count", "rbtree"],
          'rank_ratio': [0] + np.arange(0.001,1,0.005),
          'max_iter'  : [10, 20, 30],
          'timeit'    : [False],
          },
         {'alpha'     : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3],
          'optimizer' : ["PRSVM", "simple"],
          'rank_ratio': [1],
          'max_iter'  : [10, 20, 30],
          'timeit'    : [False],
          },
        ]
MSVM = MinlipSurvivalAnalysis(solver='cvxpy', 
                               alpha=1.0,
                               kernel='linear',
                               gamma=None,
                               degree=3,
                               coef0=1,
                               kernel_params=None,
                               pairs='nearest',
                               verbose=False,
                               timeit=None,
                               max_iter=None,
                               )
HLSVM = HingeLossSurvivalSVM(solver='cvxpy',
                              alpha=1.0,
                              kernel='linear',
                              gamma=None,
                              degree=3,
                              coef0=1,
                              kernel_params=None,
                              pairs='all',
                              verbose=False,
                              timeit=None,
                              max_iter=None,
                              )
NSVM = NaiveSurvivalSVM(penalty='l2',
                        loss='squared_hinge',
                        dual=True,
                        tol=0.0001,
                        alpha=1.0,
                        verbose=0,
                        random_state=None,
                        max_iter=5e4,
                        )


def RFECV_(Es):
    from sklearn.feature_selection import RFECV
    select_Sup = []
    select_Sco = []
    selector = RFECV(Es,
                    step=1,
                    cv=10,
                    n_jobs=10)
    selector = selector.fit(X_train, Y_train)
    print(selector.grid_scores_)
    select_Sco.append(selector.grid_scores_)
    select_Sup.append(selector.ranking_)
    return ( select_Sco, select_Sup )
#RFECV_(MSVM)

from sklearn.model_selection import learning_curve
def Model(cphe):
    cphe.fit(X_train, Y_train)

    if hasattr(cphe, 'coef_'):
        coe3 = cphe.coef_
    elif hasattr(cphe, 'feature_importances_'):
        coe3 = cphe.feature_importances_
    else:
        raise RuntimeError('The classifier does not expose '
                            '"coef_" or "feature_importances_" '
                            'attributes')

    pre3 = cphe.predict(X_test)
    sco_tr = cphe.score(X_train, Y_train)
    sco_te = cphe.score(X_test , Y_test)

    print('coe3', coe3)
    print(coe3.shape)
    #print('pre3', pre3)
    print('params',cphe.get_params)
    print('sco_tr', sco_tr)
    print('sco_te', sco_te)
    return (sco_tr, sco_te)
print('*'*60)
print('*'*60)
def lear_c(EST, name, params):
    tr_a = []
    te_a = []
    for i in params:
        EST.set_params( **{ name : i } )
        tr_s, te_s = Model(EST)
        tr_a.append(tr_s)
        te_a.append(te_s)

    plt.plot( params,tr_a, label='train' )
    plt.plot( params,te_a, label='test' )
    plt.legend()
    plt.show()


alpha = np.arange(0.001, 0.05, 0.001)
optimizer = ["avltree", "direct-count", "PRSVM", "rbtree", "simple"]
rank_ratio = np.arange(0.00,1,0.02)
max_iter = np.arange(1,21, 1)
#timeit =  np.arange(1,21, 1),
lear_c( HLSVM, 'alpha', alpha)

