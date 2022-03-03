
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

class Results:

  def __init__(self, numOfFiles):
    self.num = numOfFiles

    hdf = pd.read_hdf("sandbox.h5")
    files_by_auth_name = hdf.groupby(['username']).indices
    authors_with_k = dict(filter(lambda x: len(x[1]) >= self.num, files_by_auth_name.items()))
    authors_with_k = {author: authors_with_k[author] for author in list(authors_with_k)}
    rng = np.random.default_rng(1)
    for k in authors_with_k:
      authors_with_k[k] = rng.choice(authors_with_k[k], self.num, replace=False, shuffle=False)
    # List of all files sorted by author where
    #each author has exactly k files
    files = np.concatenate(list(authors_with_k.values()))

    #generate labels
    y = np.floor(np.arange(len(files)) / self.num)

    #sort data
    cross_val_indicies = np.zeros(len(files), dtype=np.int32)
    # files should always be divizable by k
    fold_size = int(len(files) / self.num)
    for i in range(self.num):
      cross_val_indicies[i * fold_size:(i + 1) * fold_size] = np.arange(i, len(files), self.num)
    fold_size = int(len(files) / self.num)
    # Reorganize labels
    y = y[cross_val_indicies]

    #begin Brian Code that trims the tfidf features

    tfidfs = sp.load_npz("tfidfs.npz").toarray()
    chosen_indicies = np.load("chosen.npy")
    X = np.zeros([cross_val_indicies.shape[0], len(chosen_indicies)])
    for i, cross_val_index in enumerate(cross_val_indicies):
      #print(i, end="\r")
      X[i, : ] = tfidfs[files[cross_val_index],chosen_indicies]

    self.data = X
    self.target = y

  def tree(self):
    rfc = RandomForestClassifier()
    cv = KFold(n_splits=self.num, shuffle=False)
    return(cross_val_score(rfc,self.data, self.target, verbose = 0, cv = cv))

  def knn(self):
    knn = KNeighborsClassifier()
    cv = KFold(n_splits=self.num, shuffle=False)
    return (cross_val_score(knn, self.data, self.target, verbose=0, cv=cv))
