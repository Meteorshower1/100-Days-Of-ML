import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

def draw(classifier,X_set,y_set,title,xlabel,ylabel):
    step = 0.01
    start_X1_set = X_set[:,0].min() - 1 #算出特征1(年纪)起始点
    stop_X1_set = X_set[:,0].max() + 1  #算出特征1结束点

    print('Start X1 {0} Stop X1 {1}'.format(start_X1_set,stop_X1_set))
    X1 = np.arange(start=start_X1_set,stop=stop_X1_set,step=step)

    print('X1:{0}'.format(X1))

    start_X2_set = X_set[:,1].min() - 1 #算出特征2起始点
    stop_X2_set = X_set[:,1].max() + 1  #算出特征2结束点

    print('Start X2 {0} Stop X2 {1}'.format(start_X2_set,stop_X2_set))

    X2 = np.arange(start=start_X2_set,stop=stop_X2_set,step=step)

    print('X2:{0}'.format(X2))

    X1,X2=np.meshgrid(X1,X2) #展开为网格数据

    print('meshgrid = X1,X2:{0} {1}'.format(X1,X2))

    DrawX1 = X1.ravel() #多维数组转换为一维数组的功能
    DrawX2 = X2.ravel() 

    InputArray = np.array([DrawX1,DrawX2]).T #特征1和特征2组成的网格点
 
    YPredict = classifier.predict(InputArray).reshape(X1.shape) #平铺所有网格结果的学习数据
    plt.contourf(X1, X2, YPredict, alpha = 0.50, cmap = ListedColormap(('red', 'green'))) #底色 0 红色， 1 绿色
    # plt.xlim(X1.min(),X1.max()) ##限制X轴坐标的最小/大值
    # plt.ylim(X2.min(),X2.max()) ##限制Y轴坐标的最小/大值
    for i,j in enumerate(np.unique(y_set)):
        tx = y_set == j
        plt.scatter(X_set[tx,0],X_set[tx,1],c = ListedColormap(('red', 'green'))(i), label=j) #绘制散列点
    plt. title(title)
    plt. xlabel(xlabel)
    plt. ylabel(ylabel)
    plt. legend()
    

dataset = pd.read_csv('datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [ 2, 3]].values     #年纪,估计收入
Y = dataset.iloc[:,4].values            #是否购买

# transformers=[
#     ("OneHotEncoder",OneHotEncoder(categories = [["Female","Male"]]),[3])
# ]
# colT = ColumnTransformer(transformers)

# X_OneHot = colT.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)


plt.subplot(2, 1, 1)
draw(classifier,X_train,y_train,' LOGISTIC(Training set)',' Age',' Estimated Salary')
plt.subplot(2, 2, 3)
draw(classifier,X_test,y_test,' LOGISTIC(Test set)',' Age',' Estimated Salary')
plt. show()
