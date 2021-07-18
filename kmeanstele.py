# 3

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Load the dataset

telecom = pd.read_excel('C:\\Users\\WIN10\\Desktop\\LEARNING\\Telco_customer_churn.xlsx')

telecom.describe()

telecom.info()

# Performing EDA to the dataset

# Checking for NA values in our dataset

telecom.isna().sum()

telecom.columns

# Plotting histogram to get the skewness 

plt.hist(telecom['Tenure in Months'],bins=10,color='blue');plt.xlabel('Tenure in Months');plt.ylabel('Frequency');plt.title('Tenure in Months vs Frequency') # Histogram is normally skewed

plt.hist(telecom['Avg Monthly Long Distance Charges'],bins=10,color='blue');plt.xlabel('Avg Monthly Long Distance Charges');plt.ylabel('Frequency');plt.title('Avg Monthly Long Distance Charges vs Frequency') # Histogram is positively skewed

plt.hist(telecom['Avg Monthly GB Download'],bins=10,color='blue');plt.xlabel('Avg Monthly GB Download');plt.ylabel('Frequency');plt.title('Avg Monthly GB Download vs Frequency') # Histogram is positively skewed

plt.hist(telecom['Monthly Charge'],bins=10,color='blue');plt.xlabel('Monthly Charge');plt.ylabel('Frequency');plt.title('Monthly Charge vs Frequency') # Histogram is positively skewed

plt.hist(telecom['Total Charges'],bins=10,color='blue');plt.xlabel('Total Charges');plt.ylabel('Frequency');plt.title('Total Charges vs Frequency') # Histogram is positively skewed

plt.hist(telecom['Total Long Distance Charges'],bins=10,color='blue');plt.xlabel('Total Long Distance Charges');plt.ylabel('Frequency');plt.title('Total Long Distance Charges vs Frequency') # Histogram is positively skewed

plt.hist(telecom['Total Revenue'],bins=10,color='blue');plt.xlabel('Total Revenue');plt.ylabel('Frequency');plt.title('Total Revenue vs Frequency') # Histogram is positively skewed

# Plotting boxplot for getting the outliers

plt.boxplot(telecom['Tenure in Months'],vert=False);plt.xlabel('Tenure in Months');plt.ylabel('frequency');plt.title('Tenure in Months vs Frequency') # There are no outliers in this column

plt.boxplot(telecom['Avg Monthly Long Distance Charges'],vert=False);plt.xlabel('Avg Monthly Long Distance Charges');plt.ylabel('frequency');plt.title('Avg Monthly Long Distance Charges vs Frequency') # There are no outliers in this column

plt.boxplot(telecom['Avg Monthly GB Download'],vert=False);plt.xlabel('Avg Monthly GB Download');plt.ylabel('Frequency');plt.title('Avg Monthly GB Download vs Frequency') # There are outliers in this column

# Replacing the outliers by maximum and minimum values

IQR1 = telecom['Avg Monthly GB Download'].quantile(0.75)-telecom['Avg Monthly GB Download'].quantile(0.25)

lowerlimit1 = telecom['Avg Monthly GB Download'].quantile(0.25)-(IQR1*1.5)

lowerlimit1

upperlimit1 = telecom['Avg Monthly GB Download'].quantile(0.75)+(IQR1*1.5)

upperlimit1

telecom['Avg Monthly GB Download'] = pd.DataFrame(np.where(telecom['Avg Monthly GB Download']>upperlimit1,upperlimit1,np.where(telecom['Avg Monthly GB Download']<lowerlimit1,lowerlimit1,telecom['Avg Monthly GB Download'])))

plt.boxplot(telecom['Avg Monthly GB Download'])

# Outliers have been replaced

plt.boxplot(telecom['Monthly Charge'],vert=False);plt.xlabel('Monthly Charge');plt.ylabel('Frequency');plt.title('Monthly Charge vs Frequency') # There are no outliers in this coulmn

plt.boxplot(telecom['Total Charges'],vert=False);plt.xlabel('Total Charges');plt.ylabel('Frequency');plt.title('Total Charges vs Frequency') # There are no outliers in this coulmn

plt.boxplot(telecom['Total Long Distance Charges'],vert=False);plt.xlabel('Total Long Distance Charges');plt.ylabel('Frequency');plt.title('Total Long Distance Charges vs Frequency') # There are outliers in this coulmn

# Replacing the outliers by maximum and minimum values

IQR2 = telecom['Total Long Distance Charges'].quantile(0.75)-telecom['Total Long Distance Charges'].quantile(0.25)

lowerlimit2 = telecom['Total Long Distance Charges'].quantile(0.25)-(IQR2*1.5)

lowerlimit2

upperlimit2 = telecom['Total Long Distance Charges'].quantile(0.75)+(IQR2*1.5)

upperlimit2

telecom['Total Long Distance Charges'] = pd.DataFrame(np.where(telecom['Total Long Distance Charges']>upperlimit2,upperlimit2,np.where(telecom['Total Long Distance Charges']<lowerlimit2,lowerlimit2,telecom['Total Long Distance Charges'])))

plt.boxplot(telecom['Total Long Distance Charges'],vert=False)

# Outliers are removed

plt.boxplot(telecom['Total Revenue'],vert=False);plt.xlabel('Total Revenue');plt.ylabel('Frequency');plt.title('Total Revenue vs Frequency') # There are outliers in this coulmn

IQR3 = telecom['Total Revenue'].quantile(0.75)-telecom['Total Revenue'].quantile(0.25)

lowerlimit3 = telecom['Total Revenue'].quantile(0.25)-(IQR3*1.5)

lowerlimit3

upperlimit3 = telecom['Total Revenue'].quantile(0.75)+(IQR3*1.5)

upperlimit3

telecom['Total Revenue'] = pd.DataFrame(np.where(telecom['Total Revenue']>upperlimit3,upperlimit3,np.where(telecom['Total Revenue']<lowerlimit3,lowerlimit3,telecom['Total Revenue'])))

plt.boxplot(telecom['Total Revenue'])

# Checking the normality of the dataset

import scipy.stats as stats

import pylab

stats.probplot(telecom['Tenure in Months'],dist='norm',plot=pylab)

# The data is normal 

stats.probplot(telecom['Avg Monthly Long Distance Charges'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(telecom['Avg Monthly GB Download'],dist='norm',plot=pylab)

# The data is non normal, so we will do transformation

stats.probplot(np.log(telecom['Avg Monthly GB Download']),dist='norm',plot=pylab)

stats.probplot(np.sqrt(telecom['Avg Monthly GB Download']),dist='norm',plot=pylab)

stats.probplot((1/telecom['Avg Monthly GB Download']),dist='norm',plot=pylab)

# Even after applying the transformations the data is non normal so we will proceed with the same data

stats.probplot(telecom['Monthly Charge'],dist='norm',plot=pylab)

# The Monthly Charge dataset is non-normal data,we have to apply transformation and check with it.

stats.probplot(np.log(telecom['Monthly Charge']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(telecom['Monthly Charge']),dist="norm",plot=pylab)

stats.probplot((1/(telecom['Monthly Charge'])),dist="norm",plot=pylab)

# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data

stats.probplot(telecom['Total Charges'],dist="norm",plot=pylab)

# the Total Charges dataset is non-normal data, we have to apply transformation and check with it.

stats.probplot(np.log(telecom['Total Charges']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(telecom['Total Charges']),dist="norm",plot=pylab)

# After applying sqrt transformation the data is converting to normal data.

telecom['Total Charges'] = (np.sqrt(telecom['Total Charges']))

stats.probplot(telecom['Total Long Distance Charges'],dist="norm",plot=pylab)

# the Total Long Distance Charges dataset is non-normal data, we have to apply transformation and check with it.

stats.probplot(np.log(telecom['Total Long Distance Charges']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(telecom['Total Long Distance Charges']),dist="norm",plot=pylab)

# After applying sqrt transformation the data is converting to normal data.

telecom['Total Long Distance Charges']=(np.sqrt(telecom['Total Long Distance Charges']))

stats.probplot(telecom['Total Revenue'],dist="norm",plot=pylab)

# the Total Revenue dataset is non-normal data, we have to apply transformation and check with it.

stats.probplot(np.log(telecom['Total Revenue']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(telecom['Total Revenue']),dist="norm",plot=pylab)

#after applying sqrt transformation the data is converting to normal data.

telecom['Total Revenue'] = (np.sqrt(telecom['Total Revenue']))    

# For the character column we have to apply dummy variable creation

telecomdummy = pd.get_dummies(telecom,columns=['Quarter','Referred a Friend','Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract','Paperless Billing','Payment Method'],

                                   drop_first=True)

# Here we are doing the dummy variable creation for the discrete data by ignoring first dummy variable.

# Here i am applying the technique if we have n model we need to create n-1 dummy variable.

# Checking for zero variance in all columns

print(np.var(telecomdummy,axis=0))

# We have zero variance in count column

# We will remove count column

telecomdummy = telecomdummy.drop(['Count'],axis=1)

telecomdummy.describe()

# We will Standardize the data to make it scale free

from sklearn.preprocessing import scale

standtele = scale(telecomdummy.iloc[:, 1:])

print(np.mean(standtele,axis=0))

# After standardizing we are getting mean as 0

print(np.var(standtele,axis=0))

# scree plot or elbow curve decide the k-value

from sklearn.cluster import	KMeans

twss=[]

k = list(range(2,9))

for i in k:

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(standtele)

    twss.append(kmeans.inertia_)

twss

# Scree plot or elbow curve decide the k-value

plt.plot(k,twss,'ro-');plt.xlabel("No of clusters");plt.ylabel("total_within_ss")

# slecting cluster

model = KMeans(n_clusters=3)

model.fit(standtele)

model.labels_

mb=pd.Series(model.labels_)

telecom['k_means_clust']=mb

telecom.head()



telecom = telecom.iloc[:, [0,30,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]

telecom.head()

# Aggregate mean of each cluster

mean1 = telecom

mean1.columns

mean1 = mean1.drop(['Customer ID', 'k_means_clust', 'Count', 'Quarter', 'Referred a Friend','Offer', 'Phone Service', 'Multiple Lines','Internet Service', 'Internet Type','Online Security', 'Online Backup', 'Device Protection Plan',

       'Premium Tech Support', 'Streaming TV', 'Streaming Movies','Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing','Payment Method'],axis=1)

mean1.iloc[:, 1:].groupby(telecom.k_means_clust).mean()

# we cannot apply mean to the character data, for only numeric data only we can apply

# so we apply mode to character data

mode1 = telecom

mode1.columns

mode1 = mode1.drop(['k_means_clust','Number of Referrals', 'Tenure in Months','Avg Monthly Long Distance Charges', 'Avg Monthly GB Download','Monthly Charge', 'Total Charges', 'Total Refunds',

       'Total Extra Data Charges', 'Total Long Distance Charges','Total Revenue'],axis=1)

mode1.iloc[:, 1:].groupby(telecom.k_means_clust).agg(pd.Series.mode)

# creating a csv file 

telecom.to_csv("KmeansTelecustomer.csv", encoding = "utf-8")

import os

os.getcwd()

