from sklearn.datasets import load_digits
digits = load_digits()

digits.data.shape
digits.target

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label,fontsize=20)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)

logisticRegr.predict(x_test[0].reshape(1,-1))
logisticRegr.predict(x_test[0:10])
pred = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test,y_test)
score

import numpy as np
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt='.3f',linewidth=.5,square=True,cmap='Blues_r')
