from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_array
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

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1])
sss = StratifiedShuffleSplit_(n_splits=5, test_size=0.5, random_state=0)
sss.get_n_splits(X, y)


from sksurv.datasets import load_veterans_lung_cancer

data_x, data_y = load_veterans_lung_cancer()
print( np.array( data_y.tolist() )[:,0] ) 
print( dir(data_y) )

for train_index, test_index in sss.split(data_x, data_y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = data_x[train_index], data_x[test_index]
   y_train, y_test = data_y[train_index], data_y[test_index]
