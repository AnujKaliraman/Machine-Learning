# Data Pre-processing template

#Importing the libraries

import numpy as np                # importing the mathematics models
import matplotlib.pyplot as mpt   # pyplot is the sub library in matplotlib
                                  # helps plot charts
                                  
import pandas as  pd                # import datasets and manage datasets


# Importing the dataset
    # First thing to do is to set the working directory. (Folder which contain data)
    # Every time importing a dataset, we have to set a working directory

dataset = pd.read_csv('Data.csv')

# Then separate the dependent and the independent variables

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# Taking care of the missing data
from sklearn.impute import SimpleImputer    # importing imputer class for missing data

# creating object for the class 

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3] )

#Categorical variables
         #encode text into numbers
         
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=np.float)

         # using the dummy variables to make sure that the there is no relation between the 
         # categories

#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

#encoding the dependent variable
#won't have to use the onehotencoder since it's a dependent variable

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

#splitting the data into the training set and the test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Feature scaling
    #variables are not on the same scale by default
    # different scale will cause the problem because the machine learning models are 
    # based on the Euclidean distance (square root of the sum of the squared difference in the distance)
    # if one variable have higher range of values then the Euclidean distance will be dominated by that
    # variable
    # ways of scaling the data: 1. Standardisation(x-mean(x)/standard deviation (x)
    #2. Normalisation (x-min(x)/max(x) - min(x))
    

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)     # we don't need to fit the test set as it is already fitted to the
                                    # training set
                                    # Do we need to scale the dummy variables?
                                    # It depends on the content: we might lose the interpretation
                                    # we don't scale the dummy variables the model does not break

# Decision trees are not based on the Euclidean distances but we still need to scale it because if we 
# don't then they will run for very long time

                                    
                                    
