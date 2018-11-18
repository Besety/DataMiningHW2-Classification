import pandas as pd
import numpy as np
from sklearn import tree
import sys
import os       
import seaborn as sns
import pydotplus 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\bin'


titanic = pd.read_csv('titanic_data.csv')
X=titanic[['Pclass','Age','Sex','Fare','SibSp','Parch','Cabin','Embarked']]
y=titanic['Survived']
X['Fare']=X['Fare'].map(lambda x:np.log(x+1))
X['Age'].fillna(X['Age'].mean(),inplace=True)
X['Age']=X['Age'].map(lambda x: 'child' if x<12 else 'youth' if x<30 else 'adlut' if x<60 else 'old' if x>60 else 'null')
X['Fare']=X['Fare'].map(lambda x: 'poor' if x<2.5 else 'rich')
X['Cabin']=X['Cabin'].map(lambda x:'yes' if type(x)==str else 'no')
X['SibSp']=X['SibSp'].map(lambda x: 'small' if x<1 else 'middle' if x<3 else 'large')
X['Parch']=X['Parch'].map(lambda x: 'small' if x<1 else 'middle' if x<4 else 'large')


#One hot encoding
X = pd.get_dummies(X)
encoded = list(X.columns)
print( "{} total features after one-hot encoding.".format(len(encoded)) )


#切Train and Test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)



vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
print(vec.feature_names_)



#Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict=dtc.predict(X_test)
print(dtc.score(X_test,y_test))
print(classification_report(y_predict,y_test,target_names=['died','survived']))
print(dtc.feature_importances_)


#Random Forest
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)
print(rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test))


#可視化 Decision Tree
with open("titanic.dot", 'w') as f:
    f = tree.export_graphviz(dtc, out_file=f)

dot_data = tree.export_graphviz(dtc, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("titanic.pdf") 
