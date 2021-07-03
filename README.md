# Classifier-prediction
all different types of classifier 
Data: The zip file “hw2.q1.data.zip” contains 3 CSV files: 
•	“hw2.q1.train.csv” contains 10,000 rows and 26 columns. The first column ‘y’ is the output variable with 2 classes: 0, 1. The remaining 25 columns contain input features: x_1, …, x_25. 
•	“hw2.q1.test.csv” contains 5,000 rows and 41 columns. The first column ‘y’ is the output variable with 2 classes: 0, 1. The remaining 25 columns contain input features: x_1, …, x_25.
•	“hw2.q1.new.csv” contains 30 rows and 26 columns. The first column ‘ID’ is an identifier for 30 unlabeled samples. The remaining 25 columns contain input features: x_1, …, x_25.

Task 1. [4 points]
Use 5-fold cross-validationwith the 10,000 labeled exampled from “hw2.q1.train.csv” to determine the fewest number of rules using which a decision tree classifier can achieve mean cross-validation accuracy of at least 0.96.  Report the number of rules needed, the cross-validation accuracy obtained, and all the hyper-parameter values for the DecisionTreeClassifier.
Number of rules needed:…7…………….
Mean cross-validation accuracy: ……0.9690…………………. (rounded to 4 decimal places)
Hyper-parameter values for selected DecisionTreeClassifier model:
[criterion='gini',
    splitter='best',
max_depth=3,
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.0,
max_features=None,
random_state=None,
max_leaf_nodes=None,
min_impurity_decrease=0.0,
min_impurity_split=None,
class_weight=None,
ccp_alpha=0.0 ]



 
Task 2. [2 Points]
Train a DecisionTreeClassifier with the hyper-parameter values determined in Task 1 on all 10,000 training samples and use it to predict the output class ‘y’ for the 2,000 examples in “hw2.q1.test.csv”. Report the following:
•	Accuracy on 2,000 test examples: …………97.1000…………  (rounded to 4 decimal places)
•	Classification report for the 2,000 test examples:

precision    recall  f1-score   support

         0.0       0.97      0.98      0.97      1048
         1.0       0.97      0.96      0.97       952

    accuracy                           0.97      2000
   macro avg       0.97      0.97      0.97      2000
weighted avg       0.97      0.97      0.97      2000











•	Of the 952 test samples that belong to class y=1, how many are correctly predicted (according to your classification report)?     918

 
Task 3. [2 Points]
Use the model trained in Task 2 to predict the output class ‘y’ for the 30 examples in “hw2.q1.new.csv”. Specify the predicted classes in the table below:

ID	predicted y
1	 0
2	 0
3	 0
4	 0
5	 0
6	 0
7	 0
8	 0
9	 0
10	 1
11	 0
12	 0
13	 0
14	 0
15	 0
16	 0
17	 0
18	 1
19	 1
20	 1
21	 1
22	 1
23	 1
24	 1
25	 1
26	 1
27	 1
28	 1
29	 1
30	 1
 
Task 4. [2 Points]
Of the 25 input variables which ones are relevant for this classification task?
The following input variables are relevant for this classification task: ……x[21],x[11],x[3],x[16],x[12]……………

Interpret your trained model and specify the rules that can be used to classify the output based on the inputs.
Rules:
Rule 1. if feature_21 <= 0.87
|   |---   if feature_11 <= 1.01
|   |   |--- if feature_3 <= 1.09
|   |   |   |--- class: 1.0

Rule 2.if feature_21 <= 0.87
|   |---   if feature_11 <= 1.01
|   |   |--- if feature_3 > 1.09
|   |   |   |--- class: 0.0

Rule 3.  if feature_21 <= 0.87
|   |---   if feature_11 > 1.0
|   |   |   |--- class: 0.0

Rule 4.  if feature_21 >  0.87
|   |---   if feature_16 <= 1.71
|   |   |--- if feature_12 <= -1.72
|   |   |   |--- class: 0.0

Rule 5.   if feature_21 >  0.87
|   |---   if feature_16 <= 1.71
|   |   |--- if feature_12 > -1.72
|   |   |   |--- class: 0.0

Rule 6.  if feature_21 >  0.87
|   |---   if feature_16 > 1.71
|   |   |   |--- class: 1.0



 
Question 2.Supervised machine learning classifiers[10 Points]
Data: The zip file “hw2.q2.data.zip” contains 3 CSV files: 
•	“hw2.q2.train.csv” contains 8,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10. 
•	“hw2.q2.test.csv” contains 2,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10.
•	“hw2.q1.new.csv” contains 30 rows and 10 columns. The first column ‘ID’ is an identifier for 30 unlabeled samples. The remaining 10 columns contain input features: x1, …, x10.

Task 1. [6 points]
Use 4-fold cross-validation with the 8,000 labeled exampled from “hw2.q2.train.csv” to identify a classifier that achieves mean cross-validation accuracy of at least 0.96.  You should try several Scikit-Learn classifiers, including: GaussianNB, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, KNeighborsClassifier, LogisticRegression, SVC, and MLPClassifier. Try different hyper-parameter values for the better performing classifiers to obtain a good set of hyper-parameter values. Then select the best performing model. Report the following:
Selected model with hyper-parameter values: 

Selected model : SVC
Hyper Parameters:
C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=False,
tol=0.001,
cache_size=200,
class_weight=None,
    verbose=False,
max_iter=-1,
decision_function_shape='ovr',
break_ties=False,
random_state=None





Mean cross-validation accuracy: …97.0000……………………. (rounded to 4 decimal places)

 
