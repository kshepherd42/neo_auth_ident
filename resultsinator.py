
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

min_files = 10

hdf = pd.read_hdf("sandbox.h5")
files_by_auth_name = hdf.groupby(['username']).indices
authors_with_k = dict(filter(lambda x: len(x[1]) >= min_files, files_by_auth_name.items()))
authors_with_k = {author: authors_with_k[author] for author in list(authors_with_k)}
rng = np.random.default_rng(1)
for k in authors_with_k:
  authors_with_k[k] = rng.choice(authors_with_k[k], min_files, replace=False, shuffle=False)
# List of all files sorted by author where
#each author has exactly k files
files = np.concatenate(list(authors_with_k.values()))

#generate labels
y = np.floor(np.arange(len(files)) / min_files)

#sort data
cross_val_indicies = np.zeros(len(files), dtype=np.int32)
# files should always be divizable by k
fold_size = int(len(files) / min_files)
for i in range(min_files):
  cross_val_indicies[i * fold_size:(i + 1) * fold_size] = np.arange(i, len(files), min_files)
fold_size = int(len(files) / min_files)
# Reorganize labels
y = y[cross_val_indicies]

#begin Brian Code that trims the tfidf features

tfidfs = sp.load_npz("tfidfs.npz").toarray()
chosen_indicies = np.load("chosen.npy")
X = np.zeros([cross_val_indicies.shape[0], len(chosen_indicies)])
for i, cross_val_index in enumerate(cross_val_indicies):
  print(i, end="\r")
  X[i, : cross_val_indicies.shape[0]] = tfidfs[files[cross_val_index],chosen_indicies]

#code to generate the results
splits = 7
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
cv = KFold(n_splits=splits, shuffle=False)
print("\nTree:")
print(cross_val_score(rfc,X, y,verbose=0, cv=cv))
print("\nKNN:")
print(cross_val_score(knn,X, y,verbose=0, cv=cv))

