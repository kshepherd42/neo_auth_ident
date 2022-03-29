from ntpath import join
from typing import Mapping
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer

#desired settings
max_features = 2000
data_h5 = 'c_cpp_h_program_cc9995_vs4048_a0.5_l-1_test.h5'
print(f"extract is now running with the following settings:\nmax features = {max_features}\ndata_h5 = {data_h5}")

vector = TfidfVectorizer(
                         max_features=max_features,
                         ngram_range=(1,3),
                         )

data = pd.read_hdf(data_h5)

#turn each list in data from [123,45,3,2,1] to ['123 45 3 2 1']
for i,lst in enumerate(data['file_content']):
    data['file_content'][i] = " ".join(np.array(data['file_content'][i]).astype(str))

tfidfs = vector.fit_transform(data['file_content'])
sp.sparse.save_npz("tfidfs.npz", tfidfs)
