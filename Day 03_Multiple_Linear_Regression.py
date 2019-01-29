import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression



dataset = pd.read_csv('datasets/50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

###将类别数据数字化

labelencoder = LabelEncoder()
Categories= list(set(X[: , 3]))
transformers=[
    ("OneHotEncoder",OneHotEncoder(categories = [Categories]),[3])
]
colT = ColumnTransformer(transformers)

X_OneHot = colT.fit_transform(X)
X = np.concatenate([X,X_OneHot],axis=1)
X = np.delete(X,[3],axis=1)

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[: , 3] = labelencoder.fit_transform(X[ : , 3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()

#躲避虚拟变量陷阱
X = X[: , 0:-1]
print(X)
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)


print(y_pred)
print("-----------------------")
print(Y_test)