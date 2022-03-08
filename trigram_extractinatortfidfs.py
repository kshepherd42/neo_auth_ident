from ntpath import join
from typing import Mapping
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer

#def dummy(data):
#    return data
#vocab = open('c_cpp_h_program_cc9995_vs4048.vocab')

vector = TfidfVectorizer(
                         #tokenizer=dummy,
                         #preprocessor=dummy,
                         ngram_range=(1,3),
                         #vocabulary=vocab,
                         #analyzer='char'
                         )

data = pd.read_hdf('sandbox.h5')

#turn each list in data from [123,45,3,2,1] to ['123 45 3 2 1']
#data['file_content', ] = " ".join(np.array(data['file_content'][i]).astype(str))
for i,lst in enumerate(data['file_content']):
    data['file_content'][i] = " ".join(np.array(data['file_content'][i]).astype(str))

tfidfs = vector.fit_transform(data['file_content'])
sp.sparse.save_npz("tfidfs.npz", tfidfs)


