# -*- coding: utf-8 -*-
"""Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o5O4b6SOt0kFJNtkSbC_1CyC_rXa8P4N

Homework 2.
##Question 1
**Decision Tree Classifier**[10 Points]

Data: The zip file “hw2.q1.data.zip” contains 3 CSV files: 
•	“hw2.q1.train.csv” contains 10,000 rows and 26 columns. The first column ‘y’ is the output variable with 2 classes: 0, 1. The remaining 25 columns contain input features: x_1, …, x_25. 
•	“hw2.q1.test.csv” contains 5,000 rows and 41 columns. The first column ‘y’ is the output variable with 2 classes: 0, 1. The remaining 25 columns contain input features: x_1, …, x_25.
•	“hw2.q1.new.csv” contains 30 rows and 26 columns. The first column ‘ID’ is an identifier for 30 unlabeled samples. The remaining 25 columns contain input features: x_1, …, x_25.

**Task 1.** [4 points]
Use 5-fold cross-validationwith the 10,000 labeled exampled from “hw2.q1.train.csv” to determine the fewest number of rules using which a decision tree classifier can achieve mean cross-validation accuracy of at least 0.96.  Report the number of rules needed, the cross-validation accuracy obtained, and all the hyper-parameter values for the DecisionTreeClassifier.
Number of rules needed:……………….
Mean cross-validation accuracy: ………………………. (rounded to 4 decimal places)
Hyper-parameter values for selected DecisionTreeClassifier model

**Task 2.** [2 Points]
Train a DecisionTreeClassifier with the hyper-parameter values determined in Task 1 on all 10,000 training samples and use it to predict the output class ‘y’ for the 2,000 examples in “hw2.q1.test.csv”. Report the following:
•	Accuracy on 2,000 test examples: ……………………  (rounded to 4 decimal places)
•	Classification report for the 2,000 test examples:
•	Of the 952 test samples that belong to class y=1, how many are correctly predicted (according to your classification report)?

**Task 3.** [2 Points]
Use the model trained in Task 2 to predict the output class ‘y’ for the 30 examples in “hw2.q1.new.csv”. Specify the predicted classes in the table below:

**Task 4.** [2 Points]
Of the 25 input variables which ones are relevant for this classification task?
The following input variables are relevant for this classification task: …………………

Interpret your trained model and specify the rules that can be used to classify the output based on the inputs.

##Qustion 2
**Data:**

 The zip file “hw2.q2.data.zip” contains 3 CSV files: 
•	“hw2.q2.train.csv” contains 8,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10. 
•	“hw2.q2.test.csv” contains 2,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10.
•	“hw2.q1.new.csv” contains 30 rows and 10 columns. The first column ‘ID’ is an identifier for 30 unlabeled samples. The remaining 10 columns contain input features: x1, …, x10.

**Task**

1.
[6 points]

Use 4-fold cross-validation with the 8,000 labeled exampled from “hw2.q2.train.csv” to identify a classifier that achieves mean cross-validation accuracy of at least 0.96.  You should try several Scikit-Learn classifiers, including: GaussianNB, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, KNeighborsClassifier, LogisticRegression, SVC, and MLPClassifier. Try different hyper-parameter values for the better performing classifiers to obtain a good set of hyper-parameter values. Then select the best performing model. Report the following:
Selected model with hyper-parameter values:

**Task**
 2. [2 Points]
Train the classifier with the hyper-parameter values determined in Task 1 on all 8,000 training samples and use it to predict the output class ‘y’ for the 2,000 examples in “hw2.q2.test.csv”. Report the following:
•	Accuracy on 2,000 test examples: ……………………  (rounded to 4 decimal places)

•	Classification report for the 2,000 test examples:

•	Of the 500 test samples that belong to class y=0, how many are correctly predicted (according to your classification report)?

**Task**
 3. [2 Points]
Use the model trained in Task 2 to predict the output class ‘y’ for the 30 examples in “hw2.q2.new.csv”.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

"""##Question 1 """

q1_new = pd.read_csv("/content/hw2.q1.new.csv")
q1_train = pd.read_csv("/content/hw2.q1.train.csv")
q1_test = pd.read_csv("/content/hw2.q1.test.csv")

q1_new.head()

q1_train.head()

q1_test.head()

X_train = q1_train.iloc[:, 1:].values
y_train = q1_train.iloc[:, 0].values
print(X_train)
print(y_train)

X_test = q1_test.iloc[:, 1:].values
y_test = q1_test.iloc[:, 0].values
print(X_test)
print(y_test)

"""##Task 1"""

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))

    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

"""the model showsd best accuracy:-

**RandomForestClassifier**
****Results****
Accuracy: 97.0500%
"""

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

"""## Feature Scaling"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

print(X_test)

"""## Training the Decision Tree Classification model on the Training set"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

"""**Predicting a new result**"""

print(classifier.predict(sc.transform([[0.631,0.574,0.672,0.767,0.734,0.614,0.689,0.621,0.953,0.499,0.411,0.153,0.735,0.172,0.171,0.512,0.622,0.027,0.403,0.942,0.066,0.947,0.661,0.258,0.785]])))

"""**Predicting the Test set results**"""

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""**Making the Confusion Matrix**"""

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""##task 2"""

# make class predictions
y_pred = classifier.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(f'ACCURACY:{accuracy}')
print('\n')
# generate classification report
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

q1_test['predicted_y']= y_pred
q1_test.head()

submission_dec = q1_test[['y', 'predicted_y']]
submission_dec

q1_new_pred=y_pred[:30]
q1_new_pred

q1_new_data = q1_new[:30]

q1_new_data['predicted_y']= q1_new_pred

q1_new_data.head()

"""##task3"""

submission= q1_new_data[['ID', 'predicted_y']]
submission

"""##task 4
Of the 25 input variables which ones are relevant for this classification task?
The following input variables are relevant for this classification task: ……x[21],x[11],x[3],x[16],x[12]……………

Interpret your trained model and specify the rules that can be used to classify the output based on the inputs.
"""

new_result= q1_new_data[['x_21','x_11','x_3','x_16','x_12']]
new_result.head()

feature_21= new_result['x_21']
feature_11=new_result['x_11']
feature_3 =new_result['x_3']
feature_16 = new_result['x_16']
feature_12 =new_result['x_12']

def prediction(feature_21,feature_11,feature_3,feature_16,feature_12):

  if feature_21 <= 0.87:
    if feature_11 <= 1.01:
      if feature_3 <= 1.09:
        print('class: 1.0')

  if feature_21 <= 0.8:
    if feature_11 <= 1.01:
      if feature_3 > 1.09:
        print('class: 0.0')

#Rule 3 
  if feature_21 <= 0.87:
    if feature_11 > 1.0:
      print('class: 0.0')

#Rule 4. 
  if feature_21 >  0.87:
    if feature_16 <= 1.71:
      if feature_12 <= -1.72:
        print('class: 0.0')

#Rule 5.
  if feature_21 >  0.87:
    if feature_16 <= 1.71:
      if feature_12 > -1.72:
        print('class: 0.0')

#Rule 6. 
  if feature_21 >  0.87:
    if feature_16 > 1.71:
      print('class: 1.0')

prediction(0.902,	0.218,	0.020,	0.571,	0.484)

prediction(0.452,	0.224,	0.861,	0.684,	0.875)

"""##qustion 2 """

q2_new = pd.read_csv("/content/hw2.q2.new.csv")
q2_train = pd.read_csv("/content/hw2.q2.train.csv")
q2_test = pd.read_csv("/content/hw2.q2.test.csv")

q2_new.head()

q2_train.head()

X_train = q2_train.iloc[:, 1:].values
y_train = q2_train.iloc[:, 0].values

print(X_train)
print(y_train)

q2_test.head()

X_test = q2_test.iloc[:, 1:].values
y_test = q2_test.iloc[:, 0].values

print(X_test)
print(y_test)

"""##Task 1"""

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))

    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

"""KNeighborsClassifier
****Results****
Accuracy: 96.9500% is show better performance

"""

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

"""## Feature Scaling"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

print(X_test)

"""## Training the K-NN model on the Training set"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

""" Predicting a new result"""

print(classifier.predict(sc.transform([[3.232,1.764,0.415,0.298,3.483,-2.967,-2.776,-0.663,-1.972,3.462]])))

""" Predicting the Test set results"""

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""##Task 2

Making the Confusion Matrix
"""

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# make class predictions
y_pred = classifier.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(f'ACCURACY:{accuracy}')
print('\n')
# generate classification report
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

new_pred=y_pred[:30]
new_pred

q2_new_data = q2_new[:30]
q2_new_data['predicted_y']= new_pred
q2_new_data.head()

"""##Task 3"""

submission = q2_new_data[['ID', 'predicted_y']]
submission