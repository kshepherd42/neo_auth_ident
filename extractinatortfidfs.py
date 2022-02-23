import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
data = pd.read_hdf('sandbox.h5')
tfidfs = vectorizer.fit_transform(data['file_content'])
fn = vectorizer.get_feature_names()
np.save("feature_names.npy",fn)
sp.sparse.save_npz("tfidfs.npz", tfidfs)



