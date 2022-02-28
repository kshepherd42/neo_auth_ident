# Review of tuning hyperparameters:

## Number of files required per author (resultsinator.py) assumes  1000 features

Number of files required per author: 7  
Number of authors: 7762  
Cross val scores (tree): [0.03065825 0.02885482 0.03336339 0.02975654 0.02795311 0.03155996 0.02975654]  
Cross val scores (KNN):  [0.02975654 0.03065825 0.03155996 0.02975654 0.02885482 0.02885482 0.02885482]  

Number of files required per author: 8  
Number of authors: 6447  
Cross val scores (tree): [0.02386117 0.02823018 0.03257329 0.03583062 0.0228013  0.03148751 0.03257329]  
Cross val scores (KNN):  [0.02386117 0.02823018 0.03257329 0.0369164  0.01954397 0.03257329 0.03040174]  

Number of files required per author: 9  
Number of authors:5084  
Cross val scores (tree): [0.02338377 0.02751032 0.03026135 0.03030303 0.0261708  0.03030303 0.0399449 ]  
Cross val scores (KNN):  [0.02200825 0.03026135 0.03026135 0.03030303 0.0261708  0.02754821 0.03719008]  

Number of files required per author: 10  
Number of authors: 4259  
Cross val scores (tree): [0.02627258 0.02791461 0.02627258 0.02463054 0.02960526 0.02960526 0.03289474]  
Cross val scores (KNN):  [0.02298851 0.02627258 0.02298851 0.02463054 0.02796053 0.02796053 0.03289474]

## Number of features selected (selectinatortfidfs.py) assuming 10 files per author

### There is seemingly a max of 4259, when I tried 5000, I got an error message saying 'could not broadcast input array from shape (5000) into shape (4260), line 42 in resultsinator.py' due to number of authors??

Number of features:  300  
Cross val scores (tree): [0.00328407 0.00164204 0.00492611 0.00328407 0.00328947 0.00328947 0.00493421]  
Cross val scores (KNN):  [0.00492611 0.00164204 0.00656814 0.00328407 0.00328947 0.00328947 0.00657895]  

Number of features:  1000  
Cross val scores (tree): [0.02627258 0.02791461 0.02627258 0.02463054 0.02960526 0.02960526 0.03289474]  
Cross val scores (KNN):  [0.02298851 0.02627258 0.02298851 0.02463054 0.02796053 0.02796053 0.03289474]

Number of features:  2000  
Cross val scores (tree):  Tree:
[0.10344828 0.11658456 0.11001642 0.0952381  0.12335526 0.10032895 0.10690789]
Cross val scores (KNN):  [0.10673235 0.10673235 0.10180624 0.10837438 0.11348684 0.09868421 0.09046053]  
