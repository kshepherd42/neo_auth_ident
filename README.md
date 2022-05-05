# neo_auth_ident

## Purpose:
- Add TFIDF features to auth_ident

## How to run:
- *should be run in the same folder as cpp_test.h5 and cpp_train.h5, likely organized_hdfs*
- Run runAll with desired parameters (none required)

## How to run riley's code:
- train_contrastive.py is the file, and it takes 3 command line arguments. type, 
exp, and combinations

## To do:
- modify the extract.py file so that it can accept train datasets (it already 
accepts test and val)
- modify the /auth_ident/models/contrastive_bilstm.py code so that the embedding 
layer doesn't do a whole lot but just accepts the tfidf features outputted by 
extract.py. 
- figure out how auth_ident params['dataset'] actually works because the 
params_dict.json file takes 3 files (test,train,val). See recent post from riley


## Notes for later:
- sandboxMaker.py is limited to 30000
