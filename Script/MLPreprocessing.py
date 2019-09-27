from sklearn.preprocessing import (label_binarize,  OneHotEncoder, FunctionTransformer,
                                    MinMaxScaler, minmax_scale, MaxAbsScaler,
                                    StandardScaler, RobustScaler, Normalizer,
                                    QuantileTransformer, PowerTransformer, OrdinalEncoder)
from sklearn.impute import SimpleImputer
from sklearn.utils._joblib import Parallel, delayed
from sklearn_pandas import DataFrameMapper, cross_val_score
from sksurv.linear_model import CoxPHSurvivalAnalysis

import joblib
import pandas as pd
import numpy as np
import os

from .MLOpenWrite import Openf, OpenM
from .MLPlots import ClusT
from .MLUnsupervising import Decomposition
from .MLNewAttribus import  CoxPHFitter_

class PreProcessing():
    def __init__(self, arg, log,  *array, score=None, model='RF', **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.score = score
        self.model = model
        self.dicts = dicts

    def Remove_Fill(self, Dfall, drop=False):
        self.log.CIF('Missing Values Processing'.center(45, '-'))
        self.log.CIF('Primary Samples and Feauters numbers: %s , %s' %Dfall.shape)
        dfa = Dfall.copy()
        if drop:
            dfa.dropna(thresh=round(dfa.shape[0]*self.arg.MissCol,0),inplace=True,axis=1)
            dfa.dropna(thresh=round(dfa.shape[1]*self.arg.MissRow,0),inplace=True,axis=0)

            for _c in dfa.columns:
                mostfreq = dfa[_c].value_counts().max()/dfa.shape[0]
                if  mostfreq > self.arg.Mtfreq:
                    dfa.dropna([_c], inplace=True,axis=1)
                    self.log.CIF('The modal number of %s is higher than %s, droped!!!' %(mostfreq, self.arg.Mtfreq) )

            self.log.CIF('Final   Samples and Feauters numbers: %s , %s' %dfa.shape)
            self.log.CIF('The Removed Samples : %s' % (list(set(Dfall.index)-set(dfa.index))) )
            self.log.CIF('The Removed Features: %s' % (list(set(Dfall.columns)-set(dfa.columns))) )

        Xa_drop = [ i for i in Dfall.columns if i in dfa.columns]
        imp = SimpleImputer(missing_values=np.nan,
                            strategy  =self.arg.MissValue,
                            fill_value=self.arg.FillValue,
                            copy=True)
        imp.fit( dfa[Xa_drop] )
        self.log.NIF('SimpleImputer paramaters:\n%s' %imp)
        self.log.CIF(45 * '-')
        return ( imp, Xa_drop )

    def Fill_Miss(self, TTdata, Xa, Ya_A, Xg, Pdata=pd.DataFrame(), drop=False):
        clf, Xa_drop  = self.Remove_Fill(TTdata[Xa], drop=drop)

        OutDF, head = (TTdata, 'TrainTest') if Pdata.empty else ( Pdata, 'Predict') 
        OutDF = OutDF[Xa_drop + Ya_A] 
        OutXY = pd.DataFrame(clf.transform( OutDF[Xa_drop] ), index=OutDF.index, columns=Xa_drop)
        OutXY[Ya_A] = OutDF[Ya_A]

        Openf( '%s%s.set.miss_fill_data.xls'%(self.arg.output, head), (OutXY)).openv()
        #ClusT( '%s%s.set.raw.person.VIF.pdf'%(self.arg.output, head)).Plot_person( OutXY, Xa_drop, Xg )
        #ClusT( '%s%s.set.raw.pair.plot.pdf'%(self.arg.output, head) ).Plot_pair( OutXY )

        return (OutXY, Xa_drop)

    def Standard_(self, dfa, scale = 'S'):
        Scalers = {
            'S' : StandardScaler(),
            'R' : RobustScaler(quantile_range=tuple(self.arg.QuantileRange)),
            'M' : MinMaxScaler(),
            'MA': MaxAbsScaler(),
            'OE': OrdinalEncoder(),
            'OH': OneHotEncoder(),
            'N' : Normalizer(),
            'QT': QuantileTransformer(),
            'PT': PowerTransformer(),
            'none' : FunctionTransformer( validate=False ),
        }
        Sca_map = [Scalers[i] for i in scale]
        Xa = list( dfa.columns )

        mapper = DataFrameMapper([ ( Xa, Sca_map ) ])
        clfit = mapper.fit( dfa )

        self.log.CIF('Standardization Pocessing'.center(45, '-'))
        self.log.NIF('Scale paramaters:\n%s' %clfit)
        self.log.CIF(45 * '-')

        return clfit

    def Standard_Feature(self, TTdata, Xa, Ya_A, Xg, Pdata=pd.DataFrame() ):
        clf  = self.Standard_(TTdata[Xa], scale= self.arg.scaler)
        Xa_F = clf.features[0][0]

        OutDF, head = (TTdata, 'TrainTest') if Pdata.empty else ( Pdata, 'Predict') 
        OutDF = OutDF[Xa_F + Ya_A] 
        OutXY = pd.DataFrame(clf.transform( OutDF[Xa_F] ), index=OutDF.index, columns=Xa_F)
        OutXY[Ya_A] = OutDF[Ya_A]

        Openf('%s%s.standard.data.xls'%(self.arg.output, head), OutXY).openv()
        ClusT('%s%s.standard.person.VIF.pdf'%(self.arg.output, head)  ).Plot_person(OutXY, Xa_F, Xg)
        #ClusT('%s%s.standard.pair.plot.pdf'%(self.arg.output, head)   ).Plot_pair(OutXY)
        #ClusT('%s%s.standard.compair.hist.pdf'%(self.arg.output, head)).Plot_hist(TTdata[Xa_F + Ya_A], OutXY, Xa_F)

        return(OutXY)

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

            #Data[_S_name +'_B'] = Data[_S_name].apply(lambda x: True if x >0 else False)
            #Y_pred = Data[[_S_name + '_B', _T_name ]].to_records(index = False)
            #cox = CoxPHSurvivalAnalysis(alpha=0.05)
            #cox.fit(Data[[_x]], Y_pred)
        Coxcoef = pd.concat(Coxcoef,axis=0)
        Openf( '%s%s.set.univarible.Cox.xls'%(self.arg.output, _T_name), (Coxcoef)).openv() 

    def MultivarCoxPH(self, Data, _X_names, _T_name, _S_name ):
        cph = CoxPHFitter_()
        cph.fit(Data[ _X_names + [_T_name, _S_name] ],
                duration_col=_T_name,
                event_col=_S_name,
                show_progress=False)
        Coxcoef = cph._summary().summary
        Coxcoef['CI_score'] = [ cph._summary().Concordance_ ] *len(_X_names)
        Openf( '%s%s.set.Multivarible.CoxPH.xls'%(self.arg.output, _T_name), (Coxcoef)).openv() 

class Engineering():
    def __init__(self, arg, log, *array, **dicts):
        self.arg = arg
        self.log = log
        self.array = array
        self.dicts = dicts
        self.arg.output = '%s/01PreProcess/%s'%( self.arg.outdir, self.arg.header )
        os.makedirs( os.path.dirname(self.arg.output), exist_ok=True)

    def Common(self):
        self.log.CIF( 'Feature Engineering'.center(45, '*') )
        (group, AYa, RYa, CYa, SYa, Xa, Xg) = OpenM(self.arg, self.log).opens()
        dfall  = OpenM(self.arg, self.log).openi()
        Xa = [i for i in dfall.columns if i in Xa]

        self.log.CIF('Group Variable Category'.center(45, '-'))
        self.log.CIF('The features numbers : %s'   % len(Xa))
        self.log.CIF('The survival labels  : \n%s' % SYa)
        self.log.CIF(45 * '-')

        _AllDF_, _Xa = PreProcessing(self.arg, self.log).Fill_Miss( dfall, Xa, AYa, Xg, drop=True)
        _AllDF  = PreProcessing(self.arg, self.log).Standard_Feature( _AllDF_, _Xa, AYa, Xg )

        ClusT(self.arg.output + 'Features.stand.MTX_complete.pdf' ).Plot_heat(_AllDF[_Xa] , _AllDF[AYa], Xg, method='complete')
        ClusT(self.arg.output + 'Features.stand.MTX_average.pdf'  ).Plot_heat(_AllDF[_Xa] , _AllDF[AYa], Xg, method='average' )

        Decomposition(self.arg, self.log).PCA( _AllDF, _Xa, AYa, self.arg.output + 'Features.stand'  )

        for sgroup, sterm in SYa.iterrows():
            _S_name = sterm['S']
            _T_name = sterm['T']
            _Xdf = _AllDF[_Xa]
            _Y_S = _AllDF[_S_name]
            _Y_T = _AllDF[_T_name]

            try:
                PreProcessing(self.arg, self.log).UnivarCoxPH(_AllDF, _Xa, _T_name, _S_name )
            except:
                pass
            try:
                PreProcessing(self.arg, self.log).MultivarCoxPH(_AllDF, _Xa, _T_name, _S_name )
            except:
                pass
            try:
                ClusT('%sFeature_%s.Regresscofes.pdf'%(self.arg.output, _T_name)).Collinearsk(_Xdf, _Y_T)
            except:
                pass

        self.log.CIF( 'Feature Engineering Finish'.center(45, '*') )
