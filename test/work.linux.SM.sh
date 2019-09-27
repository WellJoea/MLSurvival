IN=/lustre/rde/user/zhouw/00software/MLSurvivalv0.01/test
Type=$1
Sc=S
IP=$IN/Data.172.25.TrainTest.txt
GR=$IN/Data.222.25.group.txt
PR=$IN/Data.50.25.Validation.txt
#Data.50.25.Validation.txt

OU=$IN/Result
mkdir -p $OU

/lustre/rde/user/zhouw/00software/anaconda3/envs/Python3.6/bin/python \
/lustre/rde/user/zhouw/00software/anaconda3/envs/Python3.6/bin/MLsurvival.py \
    Auto -i $IP -g $GR -p $PR -o $OU -m $Type -s $Sc
#Predict -i $IP -g $GR -p $PR -o $OU -m $Type -s $Sc
#Auto -i $IP -g $GR -p $PR -o $OU -m $Type -s $Sc
#Fselect -i $IP -g $GR -o $OU -m $Type -s $Sc -kb 0.4 -st serial
#Fitting -i $IP -g $GR -o $OU -m $Type -s $Sc
#Common -i $IP -g $GR -o $OU -m $Type -s $Sc
