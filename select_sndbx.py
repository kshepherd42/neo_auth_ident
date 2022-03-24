import pandas as pd
import numpy as np
import scipy.sparse
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif



#npz = scipy.sparse.load_npz('tfidfs.npz')
#data = pd.read_hdf('sandbox.h5')
#skb = SelectKBest(score_func=chi2, k = 1000)
#X = npz
#le = LabelEncoder()
#y = le.fit_transform(data['username'])
#chosen = skb.fit_transform(X, y)
#fn = skb.get_support(True)
#print(fn[:50])
fn = range(8000)
np.save("chosen.npy", fn)
