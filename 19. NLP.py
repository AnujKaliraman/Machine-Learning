# Practice Natural Processing Lanaguage

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the tests
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
     # use set function so the algorthm can work faster.(in books or long articles)
    # when we apply stem we apply it on single word and not on entire string
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words   # Tokenization
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(max_features = 1500)
X = CV.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

'#################################################################################################'
# Logistic Reg.

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
Classifier = LogisticRegression(random_state = 0)
Classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = Classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_LR = (76+66)/200
Precision_LR = 66/(21+66)
Recall_LR = 66/(37+66)
F1_Score_LR = (2 * Precision_LR * Recall_LR) / (Precision_LR + Recall_LR)
print('Logistic Regression:','Accuracy =', Accuracy_LR, 'Precision =', Precision_LR, 'Recall =', Recall_LR, 'F1_Score =',F1_Score_LR)

'###################################################################################################'

# kernel-svm

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the training set
from sklearn.svm import SVC
Classifier = SVC(kernel = 'rbf', random_state = 0)
Classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = Classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_ksvm = (64+83)/200
Precision_ksvm = 83/(33+83)
Recall_ksvm = 83/(20+83)
F1_Score_ksvm = (2 * Precision_ksvm * Recall_ksvm) / (Precision_ksvm + Recall_ksvm)
print('kernel-svm:','Accuracy =', Accuracy_ksvm, 'Precision =', Precision_ksvm, 'Recall =', Recall_ksvm, 'F1_Score =',F1_Score_ksvm)

'####################################################################################################'

# Decision Tree Clasification

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the training set
from sklearn.tree import DecisionTreeClassifier
Classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = Classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_DT = (74+68)/200
Precision_DT = 68/(23+68)
Recall_DT= 68/(35+68)
F1_Score_DT = (2 * Precision_DT * Recall_DT) / (Precision_DT + Recall_DT)
print('Decision Tree','Accuracy =', Accuracy_DT, 'Precision =', Precision_DT, 'Recall =', Recall_DT, 'F1_Score =',F1_Score_DT)

'###############################################################################################'

# Random Forest Classification

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the training set
from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators = 50,criterion = 'entropy', random_state = 0)
Classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = Classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_RF = (84+59)/200
Precision_RF = 59/(13+59)
Recall_RF = 59/(44+59)
F1_Score_RF = (2 * Precision_RF * Recall_RF) / (Precision_RF + Recall_RF)
print('Random Forest','Accuracy =', Accuracy_RF, 'Precision =', Precision_RF, 'Recall =', Recall_RF, 'F1_Score =',F1_Score_RF)

'#################################################################################################'

# Naive Bayes

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
Classifier = GaussianNB()
Classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = Classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_NB = (55+91)/200
Precision_NB = 91/(42+91)
Recall_NB = 91/(12+91)
F1_Score_NB = (2 * Precision_NB * Recall_NB) / (Precision_NB + Recall_NB)
print('Naive Bayes','Accuracy =', Accuracy_NB, 'Precision =', Precision_NB, 'Recall =', Recall_NB, 'F1_Score =',F1_Score_NB)
