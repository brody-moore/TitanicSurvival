import csv
import pandas as pd 
from sklearn import tree

initdf = pd.read_csv('train.csv',skiprows=1, names=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
AttDF = initdf[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
SurvDF = initdf[['Survived']]

AttDF["Sex"].replace('male', 1 ,inplace=True)
AttDF["Sex"].replace('female', 0, inplace=True)
#Changed gender data to numerics for calculation in prediction later
AttDF['Embarked'].replace( 'Q', 0 , inplace=True)
AttDF['Embarked'].replace( 'C', 1, inplace=True)
AttDF['Embarked'].replace( 'S', 2, inplace=True)



AttList = AttDF.values.tolist()
SurvList = SurvDF.values.tolist()

clf = tree.DecisionTreeClassifier()

clf = clf.fit(AttList, SurvList)

prediction = clf.predict([[892,3,1,34.5,0,0,7.8292,0]])
print(prediction)