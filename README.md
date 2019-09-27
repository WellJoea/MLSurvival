# Machine Learning Survival Analysis
The [**MLSurvival**](https://github.com/WellJoea/MLSurvival.git) is based on traditional machine learning with scikit-survival and lifelines packages.<br/>
You can ues CoXPH, GBDT, SVMs in MLSurvival and implement auto standardization, feature selection, Fitting and Prediction.<br/>
___
## Installation
### Dependencies
<pre><code>     'fastcluster >= 1.1.25',
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
</code></pre>
### User installation
- download: https://github.com/WellJoea/MLSurvival.git
- cd MLSurvival
- python setup.py install
___
## useage
**MLsurvival.py -h**<br/>
**usage:** MLsurvival.py [-h] [-V] {Common,Fselect,Fitting,Predict,Utility,Auto} ...<br/>

The traditional machine learning analysis is based on sklearn package:<br/>
### **1. positional arguments:**
<p>{Common,Fselect,Fitting,Predict,Utility,Auto}</p>
<pre><code>                        machine learning models help.
    Common              The common parameters used for other models.
    Fselect             Feature selection from standardized data.
    Fitting             Fitting and predicting the training and testing
                        set from estimators.
    Predict             predict new data from fittting process model.
    Utility             the conventional analysis.
    Auto                the auto-processing: standardization, feature
                        selection, Scoring, Fitting and/or Prediction.
</code></pre>        

### **2. optional arguments:**
<pre><code>-h, --help            show this help message and exit
-V, --version         show program's version number and exit
</code></pre>

### **3. Example:**
<p>MLkit.py Auto -h</p>
<pre><code>Examples: ./test/work.SM.sh
</code></pre>       

### **4. abbreviation:**
<p>All of the estimators you can use as follows (default: CoxPH):</p>


