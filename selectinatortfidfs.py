import pandas as pd
import numpy as np
import scipy.sparse
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

npz = scipy.sparse.load_npz(sys.argv[2])
data = pd.read_hdf(sys.argv[1])
skb = SelectKBest(score_func=chi2, k = 10)
X = npz
y = data['username']
chosen = skb.fit_transform(X, y)
print(chosen)
fn = skb.get_support(True)
print(fn)
np.save("chosen.npy", fn)
