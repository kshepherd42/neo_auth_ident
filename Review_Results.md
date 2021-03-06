# Review of tuning hyperparameters:

## Number of files required per author (resultsinator.py) assumes  1000 features

- Number of files required per author: 7  
- Number of authors: 7762  
- Cross val scores (tree): [0.03065825 0.02975654 0.03336339 0.02975654 0.02795311 0.03246168 0.02975654]   
- Cross val scores (KNN):  [0.02975654 0.03065825 0.03155996 0.02975654 0.02885482 0.02885482 0.02885482]  
- Average tree cross val score: 0.03052943571428571
- Average knn cross val score: 0.029756535714285715
<br>

- Number of files required per author: 8  
- Number of authors: 6447  
- Cross val scores (tree): [0.0248139  0.03101737 0.03101737 0.03349876 0.02977667 0.02605459 0.03225806 0.03598015]  
- Cross val scores (KNN):  [0.02605459 0.02977667 0.03101737 0.03101737 0.02853598 0.02481390 0.03225806 0.03349876]  
- Average tree cross val score: 0.03055210875
- Average knn cross val score: 0.0296215875
<br>

- Number of files required per author: 9  
- Number of authors:5084  
- Cross val scores (tree): [0.02654867 0.02831858 0.03185841 0.03185841 0.03185841 0.02654867 0.0300885  0.03362832 0.03716814]  
- Cross val scores (KNN):  [0.02477876 0.02831858 0.03008850  0.03185841 0.03185841 0.0300885 0.0300885  0.03185841 0.03716814]  
- Average tree cross val score: 0.0308751233
- Average knn cross val score: 0.0306784678
<br>

- Number of files required per author: 10  
- Number of authors: 4259  
- Cross val scores (tree): [0.03286385 0.03051643 0.0258216  0.02816901 0.02816901 0.02816901 0.02816901 0.03051643 0.03286385 0.02816901]  
- Cross val scores (KNN):  [0.02816901 0.0258216  0.02816901 0.0258216  0.0258216  0.03051643 0.02816901 0.03051643 0.03286385 0.02816901]  
- Average tree cross val score: 0.029342721
- Average knn cross val score: 0.028403755

## Number of features selected (selectinatortfidfs.py) assuming 10 files per author

### There is seemingly a max of 4259, when I tried 5000, I got an error message saying 'could not broadcast input array from shape (5000) into shape (4260), line 42 in resultsinator.py' due to number of authors??

- Number of features: 300  
- Cross val scores (tree): [0.00469484 0.00234742 0.00469484 0.00469484 0.00469484 0.00469484 0.00234742 0.00469484 0.00469484 0.00469484]  
- Cross val scores (KNN):  [0.00469484 0.00234742 0.00469484 0.00469484 0.00469484 0.00469484 0.00234742 0.00469484 0.00469484 0.00469484]  
- Average tree cross val score: 0.004225356
- Average knn cross val score: 0.004225356
<br>

- Number of features: 1000  
- Cross val scores (tree): [0.03286385 0.03051643 0.0258216  0.02816901 0.02816901 0.02816901 0.02816901 0.03051643 0.03286385 0.02816901]  
- Cross val scores (KNN):  [0.02816901 0.0258216  0.02816901 0.0258216  0.0258216  0.03051643 0.02816901 0.03051643 0.03286385 0.02816901]  
- Average tree cross val score: 0.029342721
- Average knn cross val score: 0.028403755
<br>

- Number of features: 2000  
- Cross val scores (tree): [0.10798122 0.11737089 0.11971831 0.10328638 0.11502347 0.10328638 0.12676056 0.11502347 0.10093897 0.10328638]  
- Cross val scores (KNN):  [0.11267606 0.1056338  0.12206573 0.10328638 0.11032864 0.11502347 0.11971831 0.11032864 0.09859155 0.08920188]  
- Average tree cross val score: 0.111267603
- Average knn cross val score: 0.108685446
<br>

- Number of features: 4250  
- Cross val scores (tree): [0.38262911 0.35680751 0.36619718 0.33568075 0.31690141 0.3028169 0.34507042 0.32629108 0.31220657 0.3286385 ]  
- Cross val scores (KNN):  [0.31690141 0.29812207 0.30985915 0.30751174 0.28169014 0.30751174 0.28169014 0.29577465 0.27934272 0.27464789]  
- Average tree cross val score: 0.337323943
- Average knn cross val score: 0.295305165
<br>

- Number of features: 5000 (w/9 files instead of 10)  
- Cross val scores (tree): [0.36106195 0.3699115  0.37168142 0.34867257 0.37168142 0.33274336 0.36460177 0.3380531  0.34513274]  
- Cross val scores (KNN):  [0.28318584 0.28495575 0.31150442 0.31681416 0.28141593 0.29026549 0.31150442 0.27610619 0.28849558]
- Average tree cross val score: 0.35594887
- Average knn cross val score: 0.293805309