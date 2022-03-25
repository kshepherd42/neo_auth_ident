import pandas as pd
import numpy as np
import scipy.sparse
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif

#desired settings
use_score = False
numtoselect = 1000 #only matters if use_score
max_features = 5000 #only matters if !use_score
data_h5 = 'c_cpp_h_program_cc9995_vs4048_a0.5_l-1_val.h5'


if(use_score):
  npz = scipy.sparse.load_npz('tfidfs.npz')
  data = pd.read_hdf(data_h5)
  skb = SelectKBest(score_func=chi2, k = numtoselect)
  X = npz
  le = LabelEncoder()
  y = le.fit_transform(data['username'])
  chosen = skb.fit_transform(X, y)
  fn = skb.get_support(True)
else:
  fn = range(max_features)
np.save("chosen.npy", fn)
