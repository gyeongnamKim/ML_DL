import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

x = pd.read_csv('./chap03/data/credit card.csv')
x = x.drop('CUST_ID',axis=1)
x.fillna(method='ffill',inplace=True)
x.head()

scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

x_normalized = normalize(x_scaler)
x_normalized = pd.DataFrame(x_normalized)
pca = PCA(n_components=2)
x_principal = pca.fit_transform(x_normalized)
x_principal = pd.DataFrame(x_principal)
x_principal.columns = ['P1','P2']
x_principal.head()
db_default = DBSCAN(eps=0.0375,min_samples=3).fit(x_principal)
labels = db_default.labels_
colours = {}
colours[0] = 'y'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

cvec = [colours[label] for label in labels]

r = plt.scatter(x_principal['P1'],x_principal['P2'],color='y')
g = plt.scatter(x_principal['P1'],x_principal['P2'],color='g')
b = plt.scatter(x_principal['P1'],x_principal['P2'],color='b')
k = plt.scatter(x_principal['P1'],x_principal['P2'],color='k')
plt.figure(figsize=(9,9))
plt.scatter(x_principal['P1'],x_principal['P2'],c=cvec)

plt.legend((r,g,b,k),('Label 0','Label 1 ','Label 2','Label -1'))
plt.show()

db = DBSCAN(eps=0.0375,min_samples=100).fit(x_principal)
labels1 = db.labels_
colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labels1]
colours1 = ['r','g','b','c','y','m','k']
r = plt.scatter(x_principal['P1'],x_principal['P2'],marker='o',color = colours1[0])
g = plt.scatter(x_principal['P1'],x_principal['P2'],marker='o',color = colours1[1])
b = plt.scatter(x_principal['P1'],x_principal['P2'],marker='o',color = colours1[2])
c = plt.scatter(x_principal['P1'],x_principal['P2'],marker='o',color = colours1[3])
y = plt.scatter(x_principal['P1'],x_principal['P2'],marker='o',color = colours1[4])
m = plt.scatter(x_principal['P1'],x_principal['P2'],marker='o',color = colours1[5])
k = plt.scatter(x_principal['P1'],x_principal['P2'],marker='o',color = colours1[6])

plt.figure(figsize=(9,9))
plt.scatter(x_principal['P1'],x_principal['P2'],c = cvec)
plt.legend((r,g,b,c,y,m,k),('Label 0','Label 1','Label 2','Label 3','Label 4','Label 5','Label -1'),
           scatterpoints=1,
           loc='upper left',
           ncol =3, fontsize=8)
plt.show()