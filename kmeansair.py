# 1
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as stats

import pylab

# Load the dataset

airlines = pd.read_excel("C:/Users/WIN10/Desktop/LEARNING/EastWestAirlines.xlsx",sheet_name='data')

airlines.describe() # To get the summary of the dataset

airlines.info() # To get the summary of the dataset

# Performing EDA on our dataset

# Checking for NA values in our dataset

airlines.isna().sum() 

# Plotting Histogram to get the skeweness of the dataset

airlines.columns # To get the columns name

plt.hist(airlines['Balance'],color='blue',bins=10);plt.xlabel('Balance');plt.ylabel('frequency');plt.title('Balance vs frequency') # Histogram of Balance is positively skewed

plt.hist(airlines['Bonus_miles'],color='blue',bins=10);plt.xlabel('Bonus_miles');plt.ylabel('frequency');plt.title('Bonus_miles vs frequency') # Histogram of Bonus_miles is positively skewed

plt.hist(airlines['Bonus_trans'],color='blue',bins=10);plt.xlabel('Bonus_trans');plt.ylabel('frequency');plt.title('Bonus_trans vs frequency') # Histogram of Bonus_trans is positively skewed

plt.hist(airlines['Days_since_enroll'],color='blue',bins=10);plt.xlabel('Days_since_enroll');plt.ylabel('frequency');plt.title('Days_since_enroll vs frequency') #histogram of Days_since_enroll is normally skewed

# Plotting boxplot to get the outliers from the dataset

plt.boxplot(airlines['Balance'],vert=False);plt.xlabel('Balance');plt.ylabel('frequency');plt.title('Balance vs frequency') #There are outliers in the Balance dataset

# We will manually replace the  outliers value by maximum and minimum value

IQR1 = airlines['Balance'].quantile(0.75)-airlines['Balance'].quantile(0.25)

IQR1

lowerlimit1 = airlines['Balance'].quantile(0.25)-(1.5*IQR1)

lowerlimit1

upperlimit1 = airlines['Balance'].quantile(0.75)+(1.5*IQR1)

upperlimit1

# Replacing the outliers

airlines['Balance'] = pd.DataFrame(np.where(airlines['Balance']<lowerlimit1,lowerlimit1,np.where(airlines['Balance']>upperlimit1,upperlimit1,airlines['Balance'])))

plt.boxplot(airlines['Balance'],vert=False) # There are no outliers in the Balance dataset

plt.boxplot(airlines['Bonus_miles'],vert=False);plt.xlabel('Bonus_miles');plt.ylabel('frequency');plt.title('Bonus_miles vs frequency') # There are outliers in the Bonus_miles dataset

# Detection of outliers 

IQR2 = airlines['Bonus_miles'].quantile(0.75)-airlines['Bonus_miles'].quantile(0.25)

IQR2

lowerlimit2 = airlines['Bonus_miles'].quantile(0.25)-(1.5*IQR2)

lowerlimit2

upperlimit2 = airlines['Bonus_miles'].quantile(0.75)+(1.5*IQR2)

upperlimit2

# Replacing the outliers

airlines['Bonus_miles'] = pd.DataFrame(np.where(airlines['Bonus_miles']<lowerlimit2,lowerlimit2,np.where(airlines['Bonus_miles']>upperlimit2,upperlimit2,airlines['Bonus_miles'])))

plt.boxplot(airlines['Bonus_miles'],vert=False) #There are no outliers in the Bonusmiles dataset

plt.boxplot(airlines['Bonus_trans'],vert=False);plt.xlabel('Bonus_trans');plt.ylabel('frequency');plt.title('Bonus_trans vs frequency') #There are outliers in the Bonustrans dataset

IQR3 = airlines['Bonus_trans'].quantile(0.75)-airlines['Bonus_trans'].quantile(0.25)

IQR3

lowerlimit3 = airlines['Bonus_trans'].quantile(0.25)-(1.5*IQR3)

lowerlimit3

upperlimit3 = airlines['Bonus_trans'].quantile(0.75)+(1.5*IQR3)

upperlimit3

# Replacing the outliers

airlines['Bonus_trans']=pd.DataFrame(np.where(airlines['Bonus_trans']<lowerlimit3,lowerlimit3,np.where(airlines['Bonus_trans']>upperlimit3,upperlimit3,airlines['Bonus_trans'])))

plt.boxplot(airlines['Bonus_trans'],vert=False) #There are no outliers in the Bonustrans dataset

plt.boxplot(airlines['Days_since_enroll'],vert=False);plt.xlabel('Days_since_enroll');plt.ylabel('frequency');plt.title('Days_since_enroll') #There are no outliers in the murder dataset

# To check the normality of the dataset

stats.probplot(airlines['Balance'],dist="norm",plot=pylab)

# The Balance dataset is non-normal data, we have to apply transformation and check with it.

stats.probplot(np.log(airlines['Balance']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(airlines['Balance']),dist="norm",plot=pylab)

# After applying sqrt transformation data is becoming normal data

stats.probplot(airlines['Bonus_miles'],dist='norm',plot=pylab)

# the Bonusmiles dataset is non-normal data,we have to apply transformation and check with it.

stats.probplot(np.log(airlines['Bonus_miles']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(airlines['Bonus_miles']),dist="norm",plot=pylab)

stats.probplot((1/(airlines['Bonus_miles'])),dist="norm",plot=pylab)

# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data

stats.probplot(airlines['Bonus_trans'],dist='norm',plot=pylab)

# the Bonustrans dataset is non-normal data, we have to apply transformation and check with it.

stats.probplot(np.log(airlines['Bonus_trans']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(airlines['Bonus_trans']),dist="norm",plot=pylab)

stats.probplot((1/(airlines['Bonus_trans'])),dist="norm",plot=pylab)

# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data

stats.probplot(airlines['Days_since_enroll'],dist='norm',plot=pylab)

# The Days_since_enroll dataset is normal data.

# Checking the columns for zero variance

print(np.var(airlines,axis=0))

# We have variance in all the columns.

# To normalize the data we are defining a user defined function

# Normalization function 

def norm_func(i):

    x = (i-i.min())	/ (i.max()-i.min())

    return (x)

# We are dropping state columns

airlines1 = airlines.drop(['ID#'],axis=1)

airlines1 = norm_func(airlines1.iloc[:, 0:])

airlines1.describe()

# scree plot or elbow curve decide the k-value

from sklearn.cluster import	KMeans

twss=[]

k = list(range(2,9))

for i in k:

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(airlines1)

    twss.append(kmeans.inertia_)

twss

# Scree plot or elbow curve decide the k-value

plt.plot(k,twss,'ro-');plt.xlabel("No of clusters");plt.ylabel("total_within_ss")

# slecting cluster

model = KMeans(n_clusters=3)

model.fit(airlines1)

model.labels_

mb=pd.Series(model.labels_)

airlines['k_means_clust']=mb

airlines.head()

airlines = airlines.iloc[:, [0,12,1,2,3,4,5,6,7,8,9,10,11]]

airlines.head()

# Aggregate mean of each cluster

airlines.iloc[:, 1:].groupby(airlines.k_means_clust).mean()

# creating a csv file 

airlines.to_csv("Kmeansairlines.csv", encoding = "utf-8")

import os

os.getcwd()

