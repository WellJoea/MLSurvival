from sklearn.metrics import (accuracy_score, f1_score, 
                             classification_report,
                             precision_recall_curve, mean_squared_error, 
                             roc_curve, auc, r2_score, mean_absolute_error, 
                             average_precision_score, explained_variance_score)

import joblib
import os
import numpy as np
import pandas as pd

from .MLOpenWrite import OpenM, Openf
from .MLPreprocessing import PreProcessing
from .MLSupervising  import Processing
from .MLPlots import ClusT, Evaluate

class Supervised():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts

class Prediction():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.arg.output = '%s/04Prediction/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs( os.path.dirname(self.arg.output), exist_ok=True)

    def Predict_S(self, _Prediction, _pXa, _T_name, _S_name, Xg ):
        All_coefs_, All_risks_, All_AUCs_ = \
            Processing( self.arg, self.log, model=self.arg.model, Type = self.Type ).ClfP_S( _Prediction, _pXa, _T_name, _S_name )

        ### All_coefs_
        Processing( self.arg, self.log, model=self.arg.model).Coefs_S(All_coefs_, _T_name, Xg)
        ### All_risks_
        Processing( self.arg, self.log, model=self.arg.model).Score_S(All_risks_, All_coefs_, _Prediction, _T_name, _S_name, header='Predict')
        ### All_AUCs_
        Processing( self.arg, self.log, model=self.arg.model).DyROC_S(All_AUCs_, _T_name, tag='Predict')

    def SpvPredicting(self):
        self.log.CIF( 'Supervisor Predicting'.center(45, '*') )
        (group, AYa, RYa, CYa, SYa, Xa, Xg) = OpenM(self.arg, self.log).opens()
        Predt_set = OpenM(self.arg, self.log).openp()
        TTdata = Openf( '%s/01PreProcess/%sTrainTest.set.miss_fill_data.xls'%( self.arg.outdir, self.arg.header ), index_col=0).openb()

        Pr_Xa = Predt_set.columns
        TT_Xa = TTdata.columns
        Xa  = [i for i in Pr_Xa if ( (i in Xa) & (i in TT_Xa) ) ]

        _Predict, pXa = PreProcessing(self.arg, self.log).Fill_Miss( TTdata, Xa, AYa, Xg, Pdata= Predt_set, drop=False)
        _Predict      = PreProcessing(self.arg, self.log).Standard_Feature( TTdata, pXa, AYa, Xg, Pdata=_Predict )
        _Predict[AYa].fillna(np.nan, inplace=True)

        for sgroup, sterm in SYa.iterrows():
            self.Type = 'S'
            _S_name = sterm['S']
            _T_name = sterm['T']

            self.log.CIF( ('%s: Supervised Predicting'%_T_name).center(45, '*') )
            self.Predict_S( _Predict, pXa, _T_name, _S_name, Xg )
            self.log.CIF( ('%s: Supervised Predicting Finish'%_T_name).center(45, '*') )

        self.log.CIF( 'Supervisor Predicting Finish'.center(45, '*') )
