IN=/lustre/rde/user/zhouw/00software/MLSurvivalv0.01/test
Type=$1
Sc=S
IP=$IN/CRC.133.traintest.txt
GR=$IN/CRC.188.group.txt
PR=$IN/CRC.55.validation.txt
#Data.50.25.Validation.txt

OU=$IN/Result
mkdir -p $OU

MLsurvival=/lustre/rde/user/zhouw/00software/anaconda3/envs/Python3.6/bin/MLsurvival.py
$MLsurvival Auto -i $IP -g $GR -p $PR -o $OU -m $Type -s $Sc
#MLsurvival.py Fselect -i $IP -g $GR -o $OU -m $Type -s $Sc
#$MLsurvival Fitting -i $IP -g $GR -o $OU -m $Type -s $Sc -cv 6
#$MLsurvival Predict -i $IP -g $GR -p $PR -o $OU -m $Type -s $Sc
#Auto -i $IP -g $GR -p $PR -o $OU -m $Type -s $Sc
#Fselect -i $IP -g $GR -o $OU -m $Type -s $Sc -kb 0.4 -st serial
#Fitting -i $IP -g $GR -o $OU -m $Type -s $Sc
#Common -i $IP -g $GR -o $OU -m $Type -s $Sc
