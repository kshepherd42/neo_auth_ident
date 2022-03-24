from ntpath import join
from typing import Mapping
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer

#def dummy(data):
#    return data

vector = TfidfVectorizer(
                         max_features=10000,
                         ngram_range=(1,3),
                         )

data = pd.read_hdf('sandbox.h5')

#turn each list in data from [123,45,3,2,1] to ['123 45 3 2 1']
for i,lst in enumerate(data['file_content']):
    data['file_content'][i] = " ".join(np.array(data['file_content'][i]).astype(str))

tfidfs = vector.fit_transform(data['file_content'])
sp.sparse.save_npz("tfidfs.npz", tfidfs)
