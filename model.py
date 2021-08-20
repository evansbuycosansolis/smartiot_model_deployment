import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns

# read the data
df=pd.read_csv("aquaponic_data.csv")
df.head()

# shape of the data
df.shape
df.isnull().sum()
df=df.dropna()

# Select Independent Variable
X=df[["turbidity","tot_diss_solid","wqi","hardness"]

# Select Dependent Variable
y=df["sep"]


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

#Split the Data into Test (30%) and Train(70%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features=X.columns
X[features]=sc.fit_transform(X[features])

#Instantiate Classifiers
lr=LogisticRegression(random_state=42)
knn=KNeighborsClassifier()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
ada=AdaBoostClassifier()

para_knn={'n_neighbors':np.arange(1,50)}#parameters of knn
grid_knn=GridSearchCV(knn,param_grid=para_knn,cv=5)#search knn for 5 fold cross validation


#parameters for decision tree
para_dt={'criterion':['gini','entropy'],'max_depth':np.arange(1,50),'min_samples_leaf':[1,2,45,10,20,30,40,80,100]}
grid_dt=GridSearchCV(dt,param_grid=para_dt,cv=5)#grid search decision tree 5 fold cv
#gini for the giniimpurity and entropy for information gain
#min_samples_leaf:The minimum number of samplesrequired to be at a leaf node,have the effect of smoothing the model


#parameters for randomforest
#n_estimators:The number of trees in the forest
params_rf={'n_estimators':[100,200,350,500],'min_samples_leaf':[2,10,30]}
grid_rf=GridSearchCV(rf,param_grid=params_rf,cv=5)

#parameters for Adaboost
params_ada={'n_estimators':[50,100,250,400,500,600],'learning_rate':[0.2,0.5,0.8,1]}
grid_ada=GridSearchCV(ada,param_grid=params_ada,cv=5)

grid_knn.fit(X_train,y_train)
grid_dt.fit(X_train,y_train)
grid_rf.fit(X_train,y_train)
grid_ada.fit(X_train,y_train)

print("Best parameters for KNN:",grid_knn.best_params_)
print("Best parameters for Decision Tree:",grid_dt.best_params_)
print("Best parameters for RandomForest:",grid_rf.best_params_)
print("Best parameters for AdaBoost:",grid_ada.best_params_)


lr=LogisticRegression(random_state=42)
dt=DecisionTreeClassifier(criterion='entropy',max_depth=42,min_samples_leaf=1,random_state=42)
knn=KNeighborsClassifier(n_neighbors=1)
rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,random_state=42)
ada=AdaBoostClassifier(n_estimators=600,learning_rate=1)

#lets also apply bagging and boosting
bagging=BaggingClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=46,min_samples_leaf=2,random_state=42)
                          ,n_estimators=100,random_state=42)
bagging.fit(X_train,y_train)

classifiers=[('LogisticRegression',lr),('K Nearest Neighbors',knn),('Decision Tree',dt),('RandomForest',rf),('AdaBoost',ada),('BaggingClassifier',bagging)]

from sklearn.metrics import accuracy_score

for classifier_name, classifier in classifiers:
    # Fit clfto the training set
    classifier.fit(X_train, y_train)

    # predict y_pred
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    import pickle
    pickle.dump(bagging, open("model.pkl", "wb"))