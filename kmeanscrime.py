# 2
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Load the dataset

crimedata = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\crime_data.csv')

crimedata.describe()

crimedata.info()

# Performing EDA to the dataset

# Checking for NA values in our dataset

crimedata.isna().sum()

crimedata.columns

# Plotting histogram to get the skewness 

plt.hist(crimedata['Murder'],bins=10,color='blue');plt.xlabel('Murder');plt.ylabel('Frequency');plt.title('Murder vs Frequency') # Histogram is positively skewed

plt.hist(crimedata['Assault'],bins=10,color='blue');plt.xlabel('Assault');plt.ylabel('Frequency');plt.title('Assault vs Frequency') # Histogram is positively skewed

plt.hist(crimedata['UrbanPop'],bins=10,color='blue');plt.xlabel('UrbanPop');plt.ylabel('Frequency');plt.title('UrbanPop vs Frequency') # Histogram is negatively skewed

plt.hist(crimedata['Rape'],bins=10,color='blue');plt.xlabel('Rape');plt.ylabel('Frequency') # Histogram is positively skewed

# Plotting boxplot for getting the outliers

plt.boxplot(crimedata['Murder'],vert=False);plt.xlabel('Murder');plt.ylabel('frequency');plt.title('Murder vs Frequency') # There are no outliers in this column

plt.boxplot(crimedata['Assault'],vert=False);plt.xlabel('Assault');plt.ylabel('frequency');plt.title('Assalut vs Frequency') # There are no outliers in this column

plt.boxplot(crimedata['UrbanPop'],vert=False);plt.xlabel('urbanPop');plt.ylabel('Frequency');plt.title('UrbanPop vs Frequency') # There are no outliers in this column

plt.boxplot(crimedata['Rape'],vert=False);plt.xlabel('Rape');plt.ylabel('Frequency');plt.title('Rape vs Frequency') # There are outliers in this coulmn

# Replacing the outliers by maximum and minimum values

IQR1 = crimedata['Rape'].quantile(0.75)-crimedata['Rape'].quantile(0.25)

lowerlimit1 = crimedata['Rape'].quantile(0.25)-(IQR1*1.5)

lowerlimit1

upperlimit1 = crimedata['Rape'].quantile(0.75)+(IQR1*1.5)

upperlimit1

crimedata['Rape'] = pd.DataFrame(np.where(crimedata['Rape']>upperlimit1,upperlimit1,np.where(crimedata['Rape']<lowerlimit1,lowerlimit1,crimedata['Rape'])))

plt.boxplot(crimedata['Rape'],vert=False)

# Outliers are removed

# Checking the normality of the dataset

import scipy.stats as stats

import pylab

stats.probplot(crimedata['Murder'],dist='norm',plot=pylab)

# The data is normal 

stats.probplot(crimedata['Assault'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(crimedata['UrbanPop'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(crimedata['Rape'],dist='norm',plot=pylab)

# The data is normal

# Checking for zero variance in all columns

print(np.var(crimedata,axis=0))

# We have variance in all columns

# We will normalize the data to make it scale free

# For normalizing we define user defined function

def norm_func(i):

    x = (i-i.min())/(i.max()-i.min())

    return (x)

# We will drop state column since it is nominal data

crimedata1 = crimedata.drop(['Unnamed: 0'],axis=1)

crimedata1 = norm_func(crimedata1.iloc[:,1:])

crimedata1.describe()

# scree plot or elbow curve decide the k-value

from sklearn.cluster import	KMeans

twss=[]

k = list(range(2,9))

for i in k:

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(crimedata1)

    twss.append(kmeans.inertia_)

twss

# Scree plot or elbow curve decide the k-value

plt.plot(k,twss,'ro-');plt.xlabel("No of clusters");plt.ylabel("total_within_ss")

# slecting cluster

model = KMeans(n_clusters=3)

model.fit(crimedata1)

model.labels_

mb=pd.Series(model.labels_)

crimedata['k_means_clust']=mb

crimedata.head()

crimedata = crimedata.iloc[:, [5,0,1,2,3,4]]

crimedata.head()

crimedata.iloc[:, 2:].groupby(crimedata.k_means_clust).mean()

# creating a csv file 

crimedata.to_csv("Kmeanscrime.csv", encoding = "utf-8")

import os

os.getcwd()

