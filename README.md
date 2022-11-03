# Diversity-Based-Boosting-Algorithm
Implemented diversity based boosting esemble algorithm. Aim of the research was verifying quality of the algorithm from paper of Jafar Alzubi (2016).

Used to esemble: 10 decision trees with constant depth = 3. 
To compare quality of esemble classifiers some standard datasets were used (Breast Cancer, Letter Recognition, Iris, Segment, ionosphere, Statlog (Vehicle Silhouettes), Haberman’s Survival, Contraceptive Method Choice, Isolet, glass, heart-c). 
Because balanced datasets - used standard accuracy and record Win/Draw/Loss. Statistical confirmation - by test t-Student.

Results (Win/Draw/Loss; stat. significant Win/Draw/Loss):
AdaBoosting: 11/1/10; 5/12/5
Bagging: 17/1/4; 10/10/2
DivBoosting (diversity based boosting): 4/0/18; 2/10/10

Research conducted in June 2022
