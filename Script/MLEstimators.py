from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.svm import (FastKernelSurvivalSVM, FastSurvivalSVM, MinlipSurvivalAnalysis, 
                        HingeLossSurvivalSVM, NaiveSurvivalSVM)
from sksurv.nonparametric import kaplan_meier_estimator
import numpy as np
#from bartpy.sklearnmodel import SklearnModel
#from bartpy.extensions.baseestimator import ResidualBART

class CoxnetSurvivalAnalysis_(CoxnetSurvivalAnalysis):
    def fit(self, X, y):
        super(CoxnetSurvivalAnalysis_, self).fit(X, y) 
        #self.max_id = np.where( self.deviance_ratio_ == self.deviance_ratio_.max() )
        self.coefs_ = self.coef_ 
        self.coef_  = self.coefs_[:, [-1] ]
        #self.coef_  = self.coefs_[:, self.max_id[-1] ]
        return self

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

class ML():
    def __init__(self, modle, SearchCV='GSCV', *array, **dicto):
        self.modle = modle
        self.SearchCV = SearchCV
        self.array = array
        self.dicto = dicto

    def Univariation(self):
        Univariator = {
        'KM'   : {
            'estimator' : KaplanMeierFitter(),
            'parameters': {
                'GSCV'  : { 
                            },
                'RSCV'  : { 
                            }
                }},
        }
        estimator  =  Univariator[self.modle]['estimator']
        parameters =  Univariator[self.modle]['parameters'][self.SearchCV]
        return (parameters , estimator)

    def Regression(self):
        Regressor ={
        'CoxPH'   : {
            'estimator' : CoxPHSurvivalAnalysis(alpha=0.5, n_iter=100, tol=1e-09, verbose=0),
            'parameters': {
                'GSCV'  : { 'alpha'  : [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                            'n_iter' : [300, 500],
                            'ties'   : ["breslow", "efron"],
                            'tol'    : [1e-09],
                            'verbose': [0] },
                'RSCV'  : {}
                }},

        'CoxNET'   : {
            'estimator' : CoxnetSurvivalAnalysis_(n_alphas=100,
                                                 alphas=None,
                                                 alpha_min_ratio=0.0001,
                                                 l1_ratio=0.2,
                                                 penalty_factor=None,
                                                 normalize=False,
                                                 copy_X=True,
                                                 tol=1e-07,
                                                 max_iter=100000,
                                                 verbose=False,
                                                 fit_baseline_model=True,
                                                 ),
            'parameters': {
                'GSCV'  : { 'n_alphas'          : [100, 200, 300, 500, 700 ] ,
                            'alphas'            : [None ],
                            'alpha_min_ratio'   : [0.0001],
                            'l1_ratio'          : np.arange(0.001, 0.601, 0.005),
                            'penalty_factor'    : [None],
                            'normalize'         : [False],
                            'copy_X'            : [True],
                            'tol'               : [1e-07],
                            'max_iter'          : [200000],
                            'verbose'           : [False],
                            'fit_baseline_model': [True],
                         },
                'RSCV'  : {},
                }},

        'NSVM'    : {
            'estimator' :  NaiveSurvivalSVM(penalty='l2',
                                            loss='squared_hinge',
                                            dual=True,
                                            tol=0.0001,
                                            alpha=1.0,
                                            verbose=0,
                                            random_state=None,
                                            max_iter=3e4,
                            ),
            'parameters': {
                'GSCV'  : { 'penalty' : ['l2'],
                            'loss'    : ['squared_hinge', 'hinge'],
                            'dual'    : [True],
                            #'tol'     : [0.0001],
                            'tol'     : [5e-5, 1e-4, 5e-4, 5e-3, 1e-3],
                            'alpha'   : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8],
                            'max_iter': [5e4, 8e4, 1.2e5],
                            },
                'RSCV'  : {}
                }},

        'NSVMl1'    : {
            'estimator' :  NaiveSurvivalSVM(penalty='l1',
                                            loss='squared_hinge',
                                            dual=False,
                                            tol=0.0001,
                                            alpha=1.0,
                                            verbose=0,
                                            random_state=None,
                                            max_iter=3e6,
                            ),
            'parameters': {
                'GSCV'  : { 'penalty' : ['l1'],
                            'loss'    : ['squared_hinge'],
                            'dual'    : [False],
                            'tol'     : [0.0001],
                            'alpha'   : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 11], 
                            'max_iter': [1e6, 3e6, 5e6, 1.2e7],
                            },
                'RSCV'  : {}
                }},

        'FSVM'   : {
            'estimator' : FastSurvivalSVM(alpha=1,
                                          rank_ratio=1.0,
                                          fit_intercept=False,
                                          max_iter=20,
                                          verbose=False,
                                          tol=None,
                                          optimizer=None,
                                          random_state=None,
                                          timeit=False,
                                         ),
            'parameters': {
                'GSCV'  : [{'alpha'     : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3],
                            'optimizer' : ["avltree", "direct-count", "rbtree"],
                            'rank_ratio': [0] + np.arange(0.001,1,0.02),
                            'max_iter'  : [20, 40],
                            'timeit'    : [False],
                            },
                            {'alpha'     : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3],
                            'optimizer' : ["PRSVM", "simple"],
                            'rank_ratio': [1],
                            'max_iter'  : [20, 40],
                            'timeit'    : [False],
                            },
                          ],
                'RSCV'  : {}
                }},

        'FKSVM'   : {
            'estimator' : FastKernelSurvivalSVM(alpha=1,
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
                                               ),
            'parameters': {
                'GSCV'  : { 'alpha'     : [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3],
                            'optimizer' : ["avltree", "rbtree"],
                            'rank_ratio': np.arange(0, 1, 0.005),
                            'max_iter'  : [20, 30, 40],
                            'kernel'    : ["linear",  "poly", "rbf", "sigmoid", "cosine", "precomputed"],
                            'degree'    : [2,3,4,5],
                            'timeit'    : [False],
                          },
                'RSCV'  : {}
                }},

        'CGBS'    : {
            'estimator' : ComponentwiseGradientBoostingSurvivalAnalysis_(
                                loss='coxph', 
                                learning_rate=0.1, 
                                n_estimators=100, 
                                subsample=1.0, 
                                dropout_rate=0, 
                                random_state=None, 
                                verbose=0,
                            ),
            'parameters': {
                'GSCV'  : { 'loss'          : ['coxph'] ,#, 'squared', 'ipcwls'], 
                            'learning_rate' : [0.01, 0.03, 0.07, 0.1],
                            'n_estimators'  : [100, 200],
                            'subsample'     : [0.7, 0.8, 0.9],
                            }, 
                'RSCV'  : {}
                }},

        'GBS'   : {
            'estimator' : GradientBoostingSurvivalAnalysis(
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
                            ),
            'parameters': {
                'GSCV'  : { 'loss'              : ['coxph'],
                            'learning_rate'     : [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
                            'n_estimators'      : [100],
                            'min_samples_split' : [2, 3, 4, 5],
                            'min_samples_leaf'  : [1, 2, 3, 4],
                            'max_leaf_nodes'    : [None],
                            'max_depth'         : [3, 5, 7], 
                            'max_features'      : ['auto', 'sqrt', 'log2'],
                            'subsample'         : [0.7, 0.8, 0.9, 1.0], 
                            },
                'RSCV'  : {}
                }},

        }

        estimator  =  Regressor[self.modle]['estimator']
        parameters =  Regressor[self.modle]['parameters'][self.SearchCV]
        return (parameters , estimator)
