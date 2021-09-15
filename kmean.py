import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('./chap03/data/sales data.csv')
data.head()
categorical_features = ['Channel','Region']
continuous_features = ['Fresh','Milk','Grocery','Frozen','Deterngents_Paper','Delicassen']

for col in categorical_features:
    dummies = pd.get_dummies(data[col],prefix=col)
    print(dummies)
    data = pd.concat([data,dummies],axis=1)
    data.drop(col,axis=1,inplace =True)
data.head()
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    print(k)
    Sum_of_squared_distances.append(km.inertia_)
K
plt.plot(K,Sum_of_squared_distances,'bx-')