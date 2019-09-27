from sksurv.datasets import load_veterans_lung_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_x, data_y = load_veterans_lung_cancer()
time, survival_prob = kaplan_meier_estimator(data_y["Status"], data_y["Survival_in_days"])

data_x_numeric = OneHotEncoder().fit_transform(data_x)

print(data_y,4444)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import (StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, 
                                     RepeatedKFold, RepeatedStratifiedKFold, train_test_split,
                                     GridSearchCV, RandomizedSearchCV )

estimator = CoxPHSurvivalAnalysis()
print(dir(estimator))
print(dir(estimator._baseline_model))
parameters = {  'alpha'  : [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'n_iter' : [300],
                'tol'    : [1e-09],
                'verbose': [0] },

clf = GridSearchCV( estimator, parameters,
                    n_jobs=-1, 
                    cv=5,
                    scoring=None,
                    error_score = np.nan,
                    return_train_score=True,
                    refit = True,
                    iid=True)
clf.fit(data_x_numeric, data_y)
aa = pd.DataFrame(clf.cv_results_).sort_values(by='mean_test_score', ascending=False)
aa.to_csv('aa.xls',sep='\t')
print(clf.predict(data_x_numeric.iloc[:10,:]))
print(dir(clf))

print(dir(clf.best_estimator_))
print(clf.best_estimator_._baseline_model.baseline_survival_,0000000000000)
print(clf.best_estimator_._baseline_model.cum_baseline_hazard_,1212121212121)
print(clf.best_estimator_.predict_cumulative_hazard_function(data_x_numeric.iloc[:2,:]) , 1212121232222)
print(clf.best_estimator_.predict_survival_function(data_x_numeric.iloc[:2,:]), 4334543455432354 )

print(clf.score(data_x_numeric, data_y),1111111)
print(clf.best_estimator_.score(data_x_numeric, data_y),2222222222)
print(clf.best_score_,  1)

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
name_event, name_time = data_y.dtype.names

result = concordance_index_censored(data_y[name_event], data_y[name_time], clf.predict(data_x_numeric))
print(result)

#c_uno = concordance_index_ipcw(y_train, y_test, X_test)
c_uno = concordance_index_ipcw(data_y, data_y, clf.predict(data_x_numeric))
print(c_uno, 999999999999999)

risk_scores= clf.predict(data_x_numeric)

print(risk_scores)
print(np.dot(data_x_numeric, clf.best_estimator_.coef_))
print(np.exp( np.dot(data_x_numeric, clf.best_estimator_.coef_)) )


times = np.percentile(data_y[name_time], np.linspace(5, 81, 15))

# estimate performance on training data, thus use `va_y` twice.
va_auc, va_mean_auc = cumulative_dynamic_auc(data_y, data_y, clf.predict(data_x_numeric), times)

plt.plot(times, va_auc, marker="o")
plt.axhline(va_mean_auc, linestyle="--")
plt.xlabel("days from enrollment")
plt.ylabel("time-dependent AUC")
plt.grid(True)
plt.show()