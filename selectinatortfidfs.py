import pandas as pd
import numpy as np
import scipy.sparse
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

npz = scipy.sparse.load_npz('tfidfs.npz')
data = pd.read_hdf('sandbox.h5')
skb = SelectKBest(score_func=chi2, k = 300)
X = npz
y = data['username']
chosen = skb.fit_transform(X, y)
fn = skb.get_support(True)
np.save("chosen.npy", fn)
