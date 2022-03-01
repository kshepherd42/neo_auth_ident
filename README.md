# neo_auth_ident

## Purpose:
- Add TFIDF features to auth_ident

## How to run:
- *should be run in the same folder as cpp_test.h5 and cpp_train.h5, likely organized_hdfs*
- Run sandboxMaker.py to create sandbox file
- Run extractinatortfidfs.py to extract data
- Run selectinatortfidfs.py to select data
- Run resultsinator.py to get results

## To do:
- improve accuracy of results
- create script to run all of the above in a single file
- create parameters to mess with hyperparameters so they can be more easily edited and compared
- transitioning from unigrams to bigrams

## Notes for later:
- sandboxMaker.py is limited to 30000