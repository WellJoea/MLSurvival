from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from lifelines.utils import (
    _get_index,
    _to_list,
    _to_tuple,
    _to_array,
    inv_normal_cdf,
    normalize,
    qth_survival_times,
    coalesce,
    check_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation,
    check_nans_or_infs,
    StatError,
    ConvergenceWarning,
    StatisticalWarning,
    StepSizer,
    ConvergenceError,
    string_justify,
    format_p_value,
    format_floats,
    format_exp_floats,
    dataframe_interpolate_at_times,
    CensoringType,
)

class StratifiedShuffleSplit_(StratifiedShuffleSplit):
    def split(self, X, y, groups=None):
        '''
        try:
            y = check_array(y, ensure_2d=False, dtype=None)
        except ValueError:
            y =  np.array( y.tolist() )[:,0]
        '''
        y =  np.array( y.tolist() )[:,[0]]
        return super().split(X, y, groups)

class CoxPHFitter_(CoxPHFitter):
    def _summary(self, decimals=2, verbers =0, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        kwargs:
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """

        # Print information about data first
        justify = string_justify(18)
        if verbers:
            print(self)
            print("{} = '{}'".format(justify("duration col"), self.duration_col))

            if self.event_col:
                print("{} = '{}'".format(justify("event col"), self.event_col))
            if self.weights_col:
                print("{} = '{}'".format(justify("weights col"), self.weights_col))

            if self.cluster_col:
                print("{} = '{}'".format(justify("cluster col"), self.cluster_col))

            if self.robust or self.cluster_col:
                print("{} = {}".format(justify("robust variance"), True))

            if self.strata:
                print("{} = {}".format(justify("strata"), self.strata))

            if self.penalizer > 0:
                print("{} = {}".format(justify("penalizer"), self.penalizer))

            print("{} = {}".format(justify("number of subjects"), self._n_examples))
            print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
            print("{} = {:.{prec}f}".format(justify("partial log-likelihood"), self._log_likelihood, prec=decimals))
            print("{} = {}".format(justify("time fit was run"), self._time_fit_was_called))

            for k, v in kwargs.items():
                print("{} = {}\n".format(justify(k), v))

            print(end="\n")
            print("---")

            df = self.summary

            print(
                df.to_string(
                    float_format=format_floats(decimals),
                    formatters={
                        "exp(coef)": format_exp_floats(decimals),
                        "exp(coef) lower 95%": format_exp_floats(decimals),
                        "exp(coef) upper 95%": format_exp_floats(decimals),
                    },
                    columns=[
                        "coef",
                        "exp(coef)",
                        "se(coef)",
                        "coef lower 95%",
                        "coef upper 95%",
                        "exp(coef) lower 95%",
                        "exp(coef) upper 95%",
                    ],
                )
            )
            print()
            print(
                df.to_string(
                    float_format=format_floats(decimals),
                    formatters={"p": format_p_value(decimals)},
                    columns=["z", "p", "-log2(p)"],
                )
            )

            # Significance code explanation
            print("---")
            print("Concordance = {:.{prec}f}".format(self.score_, prec=decimals))

        with np.errstate(invalid="ignore", divide="ignore"):
            sr = self.log_likelihood_ratio_test()
            if verbers:
                print(
                    "Log-likelihood ratio test = {:.{prec}f} on {} df, -log2(p)={:.{prec}f}".format(
                        sr.test_statistic, sr.degrees_freedom, -np.log2(sr.p_value), prec=decimals
                    )
            )

        self.Concordance_ = self.score_
        self.log_like = [sr.test_statistic, sr.degrees_freedom, -np.log2(sr.p_value)]
        return self

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
