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

import argparse
import os

def Args():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                prefix_chars='-+',
                conflict_handler='resolve',
                description="\nThe survival analysis based on machine learning:\n",
                epilog='''\
Example: .''')

    parser.add_argument('-V','--version',action ='version',
                version='MLsurvival version 0.1')

    subparsers = parser.add_subparsers(dest="commands",
                    help='machine learning models help.')
    P_Common   = subparsers.add_parser('Common',conflict_handler='resolve', #add_help=False,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    help='The common parameters used for other models.')
    P_Common.add_argument("-i", "--input",type=str,
                    help='''the input train and test data file with dataframe format  by row(samples) x columns (features and Y). the sample column name must be Sample.
''')
    P_Common.add_argument("-g", "--group",type=str,#required=True,
                    help='''the group file tell the featues, groups and variable type, which has Variables, Group, Type columns. Only continuous and discrete variables are supported in variable type. Onehot variables is coming.''')
    P_Common.add_argument("-o", "--outdir",type=str,default=os.getcwd(),
                    help="output file dir, default=current dir.")
    P_Common.add_argument("-m", "--model",type=str, default='CoxPH',
                    help='''the model you can used for ML.
You can choose the models as follows:

''')
    P_Common.add_argument("-t", "--pool",type=int,default=20,
                    help="the CPU numbers that can be used.")
    P_Common.add_argument("-sc", "--SearchCV",type=str,default='GSCV',choices=[ 'GSCV','RSCV'],
                    help="the hyperparameters optimization method.")
    P_Common.add_argument("-nt", "--n_iter", type=int, default= 2500,
                    help="Number of parameter settings that are sampled in RSCV. n_iter trades off runtime vs quality of the solution.")
    P_Common.add_argument("-mr", "--MissRow",type=float,default=0.8,
                    help="The rows missvalues rate, if high than the 1-value, the rows will be removed.")
    P_Common.add_argument("-mc", "--MissCol",type=float,default=0.8,
                    help="The columns missvalues rate, if high than the 1-value, the columns will be removed.")
    P_Common.add_argument("-mq", "--Mtfreq",type=float, default=0.97,
                    help="The columns mode number frequency, if high than the 1-value, the columns will be removed."
                         "If the feature matrix is sparse, plase set a higher vales, such as the maximum 1.")
    P_Common.add_argument("-mv", "--MissValue", type=str,default='median', choices=['mean', 'median', 'most_frequent', 'constant'],
                    help="Imputation transformer for completing missing values.")
    P_Common.add_argument("-fv", "--FillValue", default=None,
                    help='''When MissValue == 'constant, fill_value is used to replace all occurrences of missing_values. Idefault fill_value will be 0 when imputing numerical data and missing_value for strings or object data types.''')
    P_Common.add_argument("-pp", "--PairPlot", action='store_true', default=False,
                    help='''the pairplot of pairwise features, it is not recommended when the number of features is large 20.''')
    P_Common.add_argument("-nj", "--n_job", type=int,default=-1,
                    help="Number of cores to run in parallel while fitting across folds.")
    P_Common.add_argument("-vm", "--CVmodel", type=str,default='SSS',
                    help="the cross validation model in GridsearchCV, RFECV and SFSCV: you can use StratifiedShuffleSplit(SSS) and LeaveOneOut(LOU) model.")
    P_Common.add_argument("-cm", "--CVfit", type=str,default='SSA',
                    help="the cross validation model: you can use StratifiedShuffleSplit(SSS), StratifiedKFold(SKF), StratifiedShuffleSplit_add(SSA) and LeaveOneOut(LOU) model.")
    P_Common.add_argument("-s", "--scaler", nargs='+', default = ['S'],
                    help='''the feature standardization, you can chose: RobustScaler(R), StandardScaler(S),MinMaxScaler(M) or not do (N).''')
    P_Common.add_argument("-qr", "--QuantileRange", type=int, nargs='+', default=[10,90],
                    help="Quantile range used to calculate scale when use the RobustScaler method.")
    P_Common.add_argument("-pt", "--pcathreshold", type=float, default=0.95,
                    help='''the threshold value of sum of explained variance ratio use for pca plot and ML training and testing.''')
    P_Common.add_argument("-qr", "--ScorePoint", type=int, nargs='+', default=[10,90],
                    help="to explort the best split point for two group survival curve in the specified quantile range .")

    P_fselect  = subparsers.add_parser('Fselect', conflict_handler='resolve', add_help=False)
    P_fselect.add_argument("-sb", "--SelectB", nargs='*', default=['ANVF', 'MI', 'RS', 'MWU', 'TTI', 'PEAS', 'SPM', 'KDT', 'LR' ],
                    help='''use statistic method to select the top highest scores features with SelectKBest.
You can choose the models as follows:
classification:......++++++++++++++++++++++
**VTh..................VarianceThreshold
**ANVF.................f_classif
**Chi2.................chi2
**MI...................mutual_info_classif
**WC...................wilcoxon
**RS...................ranksums
**MWU..................mannwhitneyu
**TTI..................ttest_ind
Regressioin:.........++++++++++++++++++++++
**VTh..................VarianceThreshold
**ANVF.................f_regression
**PEAS  ...............pearsonr
**MI...................mutual_info_classif
''')
    P_fselect.add_argument("-kb", "--SelectK", type=float, default=0.8,
                    help="the SelectKBest feature selection K best number,you can use int or float.")
    P_fselect.add_argument("-kr", "--SelectR", type=float, default=0.15,
                    help="the SelectKBest True Ratio in all statistic methods in SelectB paremetors.")
    P_fselect.add_argument("-rf", "--RFECV", action='store_false' , default=True,
                    help='''whether use RFECV method to select features.''')
    P_fselect.add_argument("-sf", "--SFSCV", action='store_true' , default=False,
                    help='''whether use SFSCV method to select features. SFSCV is based on mlxtend SequentialFeatureSelector package.''')
    P_fselect.add_argument("-sm", "--specifM", type=str, nargs='*', default=[], 
                    help="the specifed models use for feature selection instead of RFECV or/and SFSCV default model. ")
    P_fselect.add_argument("-st", "--set", type=str, default='parallel', choices=['parallel','serial'],
                    help="use serial or parallel set decese multiple specifed models. if parallel, the final the features threshold ratio is deceded by the max values in all rates.")
    P_fselect.add_argument("-sp", "--SelectCV_rep",type=int,default=1,
                    help="the repetition times for using RFECV or SFSCV in one split set.")
    P_fselect.add_argument("-up", "--UniP",type=float,default=0.1,
                    help="keep the threshold p values in Univariebel CoxPH.")
    P_fselect.add_argument("-ci", "--CIs",type=float,default=0.5,
                    help="drop the threshold CI score values in Univariebel CoxPH.")
    P_fselect.add_argument("-rr", "--RFE_rate", type=float, default=20,
                    help="the features threshold ratio selected by RFECV models. value >1: features number; 0 <value< = 1: features number rate; value=0: only remove the non-zero features.")
    P_fselect.add_argument("-sr", "--SFS_rate", type=float, default=0.3,
                    help="the features threshold ratio selected by SFSCV models. value >1: features number; 0 <value< = 1: features number rate; value=0: only remove the non-zero features.")
    P_fselect.add_argument("-kf", "--k_features", nargs='+', default = ['best'], 
                    help='''Number of features to select. the string 'best' or 'parsimonious' and a tuple containing a min and max ratio can be provided, sush as [0.2, 0.75]. "best":  the feature subset with the best cross-validation performance. "parsimonious" : the smallest feature subset that is within one standard error of the cross-validation performance will be selected.''')
    P_Fselect  = subparsers.add_parser('Fselect',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_fselect],
                    help='Feature selection from standardized data.')

    P_fitting  = subparsers.add_parser('Fitting',conflict_handler='resolve', add_help=False)
    P_fitting.add_argument("-tz", "--testS",type=float,default=0.25,
                    help="the test size for cross validation when using StratifiedShuffleSplit.")
    P_fitting.add_argument("-cv", "--crossV",type=int,default=10,
                    help="the cross validation times when using StratifiedShuffleSplit.")
    P_fitting.add_argument("-pc", "--pca", action='store_true' , default=False,
                    help='''whether use pca matrix as final set for trainning as testing.''')
    P_fitting.add_argument("-lr", "--LRmode", type=str ,default='LRCV',
                    help='''use LR or LRCV model to score in RF, GBDT and XGB.''')
    P_Fitting  = subparsers.add_parser('Fitting',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_fitting],
                    help='Fitting and predicting the training and testing set from estimators.')

    P_predict  = subparsers.add_parser('Predict', conflict_handler='resolve',add_help=False,)
    P_predict.add_argument("-p", "--predict",type=str,
                    help="the predict matrix.")
    P_predict.add_argument("-ph", "--predicthead",type=str,
                    help="the predict path header.")
    P_Predict  = subparsers.add_parser('Predict',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_predict],
                    help='predict new data from fittting process model.')

    P_Autopipe = subparsers.add_parser('Auto', conflict_handler='resolve', prefix_chars='-+',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fselect, P_fitting, P_predict],
                    help='the auto-processing for all: standardization, feature selection, Fitting and/or Prediction.')
    P_Autopipe.add_argument("+P", "++pipeline",nargs='+',
                    help="the auto-processing: standardization, feature selection, Fitting and/or Prediction.")
    P_Autopipe.add_argument('+M','++MODEL' , nargs='+', type=str, default=['Standard'],
                    help='''Chose more the one models from Standard, Fselect,Fitting and Predict used for DIY pipline.''')

    args  = parser.parse_args()
    return args
