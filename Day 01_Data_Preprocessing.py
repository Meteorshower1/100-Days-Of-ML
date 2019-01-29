#Step 1: Importing the libraries
import numpy as np
import pandas as pd

#Step 2: Importing dataset
dataset = pd.read_csv('datasets/Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

print("Step 2: Importing dataset")
print("X")
print(X)
print("Y")
print(Y)

#Step 3: Handling the missing data
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values = np.nan, strategy = "mean")
Fit = imp_mean.fit(X[:,1:3])
X[:,1:3] = imp_mean.transform(X[:,1:3])
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print(X)

#Step 4: Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
print("---------------------")
print("LabelEncoder:")
print(X)
#Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print("X_test")
print(X_test)

print("X_train")
print(X_train)
print("---------------------")
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("X_test")
print(X_test)

print("X_train")
print(X_train)
