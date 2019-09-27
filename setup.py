#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
***********************************************************
* Author : Zhou Wei                                       *
* Date   : Wed Jul 10 16:57:43 2019                       *
* E-mail : welljoea@gmail.com                             *
* You are using the program scripted by Zhou Wei.         *
* The MLsurvival is used for survival analysis based on   *
* lifelines and scikit-survival.                          *
* You can implement standardization, feature selection,   *
* Fitting and Prediction.                                 *
* If you find some bugs, please send emails to me.        *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''
import sys
from os.path import dirname, join
from glob import glob
from setuptools import setup, find_packages

setup_args = {}

# Dependencies for easy_install and pip:
install_requires=[
        'fastcluster >= 1.1.25',
        'joblib >= 0.13.2',
        'lifelines >= 0.22.2',
        'matplotlib >= 3.1.1',
        'numpy >= 1.16.4',
        'pandas >= 0.24.2',
        'scikit-survival >= 0.9',
        'scipy >= 1.2.1',
        'seaborn >= 0.9.0',
        'sklearn-pandas >= 1.8.0',
        'statsmodels >= 0.10.1',
]

DIR = (dirname(__file__) or '.')

setup_args.update(
    name='MLsurvival',
    version='0.0.1',
    description=__doc__,
    author='Wei Zhou',
    author_email='welljoea@gmail.com',
    maintainer='Wei Zhou',
    #license = "MIT Licence", 
    url='https://github.com/WellJoea/MLsurvival',

    packages = find_packages(where='.', exclude=(), include=('*',)),
    include_package_data = True,

    platforms = "any",
    scripts=[join(DIR, 'MLsurvival.py')] + glob(join(DIR, 'Scripts/*.py')),
    #scripts = [],  
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
    ],

    #entry_points = {  
    #     'console_scripts': [  
    #         'test = test.help:main'  
    #     ]  
    #}
)

setup(**setup_args)
