import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

npz = sp.load_npz('tfidfs.npz')
data = pd.read_hdf('c_cpp_h_program_cc9995_vs4048_a0.5_l-1_val.h5')
skb = SelectKBest(score_func=mutual_info_regression, k = 8000)
X = npz
le = LabelEncoder()
y = le.fit_transform(data['username'])
print(y)
#y = sp.csr_matrix(y)
chosen = skb.fit_transform(X, y, )
fn = skb.get_support(True)
np.save("chosen.npy", fn)

