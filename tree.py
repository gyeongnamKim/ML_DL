import pandas as pd
df = pd.read_csv('./chap03/data/train.csv',index_col='PassengerId')
df.head()
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']]
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df = df.dropna()
x = df.drop('Survived',axis=1)
y = df['Survived']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)

from sklearn.metrics import confusion_matrix
pd.DataFrame(
    confusion_matrix(y_test,y_predict),
    columns=['Predicted Not Survival','Predicted Survival'],
    index=['True Not Survival','True Survival']
)