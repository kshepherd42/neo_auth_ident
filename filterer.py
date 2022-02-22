
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier



def crop(file_indx, crop_length, hdf):
  #""
  #Return a crop from the file at the provided index. If
  #crop_length is longer than the length of the file, then the entire
  #file will be returned.
  #""
  contents = hdf['file_content'][file_indx]
  # Minus two to account for bos and eso tokens
  max_crop = min(len(contents), crop_length) - 1
  cropped_contents = contents[:max_crop]
  return cropped_contents

def add_end_tokens(cropped_contents):
  cropped_contents = cropped_contents[:-1] + '2'
  cropped_contents = np.insert(cropped_contents, 0, 2)
  return cropped_contents




hdf = pd.read_hdf("cpp_train.h5")
files_by_auth_name = hdf.groupby(['username']).indices
authors_with_k = dict(filter(lambda x: len(x[1]) >= 7, files_by_auth_name.items()))
authors_with_k = {author: authors_with_k[author] for author in list(authors_with_k)}
#print(authors_with_k)
rng = np.random.default_rng(1)
for k in authors_with_k:
  authors_with_k[k] = rng.choice(authors_with_k[k], 7, replace=False, shuffle=False)

# List of all files sorted by author where
#each author has exactly k files
files = np.concatenate(list(authors_with_k.values()))

#generate labels
y = np.floor(np.arange(len(files)) / 7)

#sort data
cross_val_indicies = np.zeros(len(files), dtype=np.int32)
# files should always be divizable by k
print("folding")
fold_size = int(len(files) / 7)
for i in range(7):
  cross_val_indicies[i * fold_size:(i + 1) * fold_size] = np.arange(i, len(files), 7)
fold_size = int(len(files) / 7)
for i in range(7):
  cross_val_indicies[i * fold_size:(i + 1) * fold_size] = np.arange(i, len(files), 7)

print("reorganizing")
# Reorganize labels
y = y[cross_val_indicies]
# crop = np.vectorize(self.random_crop)
# files = crop(files, self.crop_length, df)
print("encoding")

X = []
#X = np.empty([cross_val_indicies.shape[0], 1000],dtype=str) #I had to do this, this is weird
#X = np.zeros([cross_val_indicies.shape[0], 1000])
print(cross_val_indicies)
print(f"len encoding: {cross_val_indicies.shape}")
for i, cross_val_index in enumerate(cross_val_indicies):
  print(i, end="\r")
  cropped = crop(files[cross_val_index], 1000, hdf)
  #print(np.array(add_end_tokens(cropped)))
  #X[i] = add_end_tokens(cropped)
  print(np.array(cropped))
  X.append(np.array(cropped))
  
  #X[i] = hdf['file_content'][i]
#print(X)
#print(X[0])



#rfc = RandomForestClassifier()
#knn = KNeighborsClassifier()
#cv = KFold(n_splits=7, shuffle=False)
#print(cross_val_score(knn,X, y,verbose=0, cv=cv))

#np.save("authors_with_k.npy",np.array(list(authors_with_k.keys())))


#print(authors_with_k)

