import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

input = "HCC.survival.192.data.txt"
input = "HCC.classify.174.data.txt"
valdR = 0.3
sampleN = 'Sample'
S = 'OS_status'
T = 'OS_months'
C = '1_year_progression'

inpudf=pd.read_csv(input, sep="\t", header=0)
def Classify():
    X = inpudf.drop([C], axis=1)
    y = inpudf[C]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=10 )
    X_tr.columns = X.columns
    X_te.columns = X.columns
    y_tr.name    = C
    y_te.name    = C

    TTS = pd.concat( (X_tr, y_tr),axis=1 )[inpudf.columns]
    PRS = pd.concat( (X_te, y_te),axis=1 )[inpudf.columns]

    outTT ="HCC.classify.%s.traintest.txt"%TTS.shape[0]
    outPR ="HCC.classify.%s.validation.txt"%PRS.shape[0]

    TTS.to_csv(outTT, sep='\t', index=False, header=True)
    PRS.to_csv(outPR, sep='\t', index=False, header=True)

def Survial():

    S0 = inpudf[ (inpudf[S]==0) ].sort_values(by=[T, S], axis=0).reset_index(drop=True)
    S1 = inpudf[ (inpudf[S]==1) ].sort_values(by=[T, S], axis=0).reset_index(drop=True)

    SS0p = np.linspace(1, S0.shape[0]-2, int(S0.shape[0] * valdR) ).astype(int)
    SS1p = np.linspace(1, S1.shape[0]-2, int(S1.shape[0] * valdR) ).astype(int)

    TTS0 = S0.loc[ set(range(S0.shape[0]) ) - set(SS0p), ]
    PRS0 = S0.loc[ SS0p, ]

    TTS1 = S1.loc[ set(range(S1.shape[0]) ) - set(SS1p), ]
    PRS1 = S1.loc[ SS1p, ]

    TTS = pd.concat((TTS0, TTS1))
    PRS = pd.concat((PRS0, PRS1))

    TTS = inpudf[ inpudf[sampleN].isin( TTS[sampleN] ) ]
    PRS = inpudf[ inpudf[sampleN].isin( PRS[sampleN] ) ]

    outTT ="HCC.survival.%s.traintest.txt"%TTS.shape[0]
    outPR ="HCC.survival.%s.validation.txt"%PRS.shape[0]

    TTS.to_csv(outTT, sep='\t', index=False, header=True)
    PRS.to_csv(outPR, sep='\t', index=False, header=True)
Classify()
