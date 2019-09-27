#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
import os
import time

from .MLPipeline import Pipeline
from .MLLogging import Logger
from .MLArguments import Args

info = '''
***********************************************************
* Author : Zhou Wei                                       *
* Date   : %s                       *
* E-mail : welljoea@gmail.com                             *
* You are using MLkit scripted by Zhou Wei.               *
* If you find some bugs, please email to me.              *
* Please let me know and acknowledge in your publication. *
* Sincerely                                               *
* Best wishes!                                            *
***********************************************************
'''%(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))


def Commands():
    args = Args()
    args.header = '/%s_%s_'%(args.model,  ''.join(args.scaler) )
    args.outdir = '%s/%s/'%(args.outdir, args.model)
    args.output = '%s%s'%(args.outdir, args.header )
    os.makedirs( args.outdir, exist_ok=True)

    Log = Logger( '%s%s_log.log'%(args.output, args.commands))
    Log.NIF(info.strip())
    Log.NIF("The argument you have set as follows:".center(59, '*'))
    for i,k in enumerate(vars(args),start=1):
        Log.NIF('**%s|%-13s: %s'%(str(i).zfill(2), k, str(getattr(args, k))) )
    Log.NIF(59 * '*')

    try:
        Pipeline(args, Log).Pipe()
        Log.CIF('Success!!!')
    except Exception as e:
        traceback.print_exc()
    finally:
        Log.CIF('You can check your progress in log file.')

