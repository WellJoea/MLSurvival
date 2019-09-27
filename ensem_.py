
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

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis



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

import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
elif platform.system()=='Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import (StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, 
                                     RepeatedKFold, RepeatedStratifiedKFold, train_test_split,
                                     GridSearchCV, RandomizedSearchCV )
                                
from sklearn.utils import safe_sqr

CGBS = ComponentwiseGradientBoostingSurvivalAnalysis(
        loss='coxph', 
        learning_rate=0.1, 
        n_estimators=100, 
        subsample=1.0, 
        dropout_rate=0, 
        random_state=None, 
        verbose=0,
        )
CGBSP = {
    'loss' : ['coxph'] ,#, 'squared', 'ipcwls'], 
    'learning_rate' : [0.01, 0.03, 0.07, 0.1],
    'n_estimators'  : [100, 200],
    'subsample'     : [0.7, 0.8, 0.9],
}
GBS  = GradientBoostingSurvivalAnalysis(
        loss='coxph',
        learning_rate=0.1,
        n_estimators=100,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_split=None,
        min_impurity_decrease=0.0,
        random_state=None,
        max_features=None,
        max_leaf_nodes=None,
        presort='auto',
        subsample=1.0,
        dropout_rate=0.0,
        verbose=0
        )
GBSP ={
    'loss'              : ['coxph'],
    'learning_rate'     : [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
    'n_estimators'      : [100],
    'min_samples_split' : [2, 3, 4, 5],
    'min_samples_leaf'  : [1, 2, 3, 4],
    'max_leaf_nodes'    : [None],
    'max_depth'         : [3, 5, 7], 
    'max_features'      : ['auto', 'sqrt', 'log2'],
    'subsample'         : [0.7, 0.8, 0.9, 1.0], 
}

class ComponentwiseGradientBoostingSurvivalAnalysis_(ComponentwiseGradientBoostingSurvivalAnalysis):
    @property
    def coef_(self):
        """Return the aggregated coefficients.

        Returns
        -------
        coef_ : ndarray, shape = (n_features + 1,)
            Coefficients of features. The first element denotes the intercept.
        """
        import numpy
        super(ComponentwiseGradientBoostingSurvivalAnalysis) #       super(CoxnetSurvivalAnalysis_, self).fit(X, y) 
        coef = numpy.zeros(self.n_features_ + 1, dtype=float)

        for estimator in self.estimators_:
            coef[estimator.component] += self.learning_rate * estimator.coef_

        return coef[1:]

CGBS_ = ComponentwiseGradientBoostingSurvivalAnalysis_(
    loss='coxph', 
    learning_rate=0.1, 
    n_estimators=100, 
    subsample=1.0, 
    dropout_rate=0, 
    random_state=None, 
    verbose=0,
    )

def Model(cphe):
    cphe.fit(DF[_Xa], Y_df)

    if hasattr(cphe, 'coef_'):
        coe3 = cphe.coef_
    elif hasattr(cphe, 'feature_importances_'):
        coe3 = cphe.feature_importances_
    else:
        raise RuntimeError('The classifier does not expose '
                           '"coef_" or "feature_importances_" '
                           'attributes')

    pre3 = cphe.predict(DF[_Xa])
    sco3 = cphe.score(DF[_Xa], Y_df)

    if hasattr(cphe, 'staged_predict'):
        staged_predict  = cphe.staged_predict(DF[_Xa]) 
        staged_predict = pd.DataFrame(staged_predict,columns =DF.index).T
        staged_predict['pre'] = pre3
        staged_predict.sort_values(by = 'pre', ascending=True, inplace =True)
        staged_predict.plot()
        plt.legend(ncol=5)
        plt.savefig('aa.pdf')
        #plt.show()

    print('coe3', coe3)
    print('coe3len', len(coe3) )
    print('pre3', pre3)
    print('sco3', sco3)
Model(CGBS_)
print('*'*60)
Model(CGBS)
#Model(GBS)

def HModel(cphe):
    cphe.fit(DF[_Xa], Y_df)

    if hasattr(cphe.best_estimator_, 'coef_'):
        coe3 = cphe.best_estimator_.coef_
    elif hasattr(cphe, 'feature_importances_'):
        coe3 = cphe.best_estimator_.feature_importances_
    else:
        raise RuntimeError('The classifier does not expose '
                           '"coef_" or "feature_importances_" '
                           'attributes')

    pre3 = cphe.predict(DF[_Xa])
    sco3 = cphe.best_estimator_.score(DF[_Xa], Y_df)

    if hasattr(cphe, 'staged_predict'):
        staged_predict  = cphe.staged_predict(DF[_Xa]) 
        staged_predict = pd.DataFrame(staged_predict,columns =DF.index).T
        staged_predict['pre'] = pre3
        staged_predict.sort_values(by = 'pre', ascending=True, inplace =True)
        staged_predict.plot()
        plt.legend(ncol=5)
        plt.savefig('aa.pdf')
        #plt.show()

    print('parameters', cphe.best_params_)
    print('coe3', coe3)
    print('pre3', pre3)
    print('sco3', sco3)

def GModel(e, p):
    glf = GridSearchCV( e , p ,
                n_jobs=-1, 
                cv=10,
                scoring=None,
                error_score = np.nan,
                return_train_score=True,
                refit = True,
                iid=True)
    Model(glf)
#GModel(CGBS, CGBSP)
