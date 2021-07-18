# 4

import numpy as np

import pandas as pd

# Load the dataset

autoinsu = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\AutoInsurance.csv')

autoinsu.describe() # To get the summary of the dataset

autoinsu.info() # To get the summary of the dataset

# Performing EDA on our dataset

# Checking for NA values in our dataset

autoinsu.isna().sum() 

# There are no NA values in our dataset

# Plotting histogram to get the skewness of the dataset

autoinsu.columns #to get the columns name

import matplotlib.pyplot as plt 

plt.hist(autoinsu['Customer Lifetime Value'],color='blue',bins=10);plt.xlabel('Customer Lifetime Value');plt.ylabel('frequency');plt.title('Customer Lifetime Value vs frequency') # histogram of Customer.Lifetime.Value is positively skewed

plt.hist(autoinsu['Income'],color='blue',bins=10);plt.xlabel('Income');plt.ylabel('frequency');plt.title('Income vs frequency') # histogram of Income is positively skewed.

plt.hist(autoinsu['Monthly Premium Auto'],color='blue',bins=10);plt.xlabel('Monthly Premium Auto');plt.ylabel('frequency');plt.title('Monthly Premium Auto vs frequency') #histogram of Monthly Premium Auto is positively skewed.

plt.hist(autoinsu['Months Since Last Claim'],color='blue',bins=10);plt.xlabel('Months Since Last Claim');plt.ylabel('frequency');plt.title('Months Since Last Claim vs frequency') #histogram of Months Since Last.Claim is normally skewed.

plt.hist(autoinsu['Months Since Policy Inception'],color='blue',bins=10);plt.xlabel('Months Since Policy Inception');plt.ylabel('frequency');plt.title('Months Since Policy Inception vs frequency') #histogram of Months Since Policy Inception is normally skewed.

plt.hist(autoinsu['Number of Policies'],color='blue',bins=10);plt.xlabel('Number of Policies');plt.ylabel('frequency');plt.title('Number of Policies vs frequency') #histogram of Number of Policies is positively skewed.

plt.hist(autoinsu['Total Claim Amount'],color='blue',bins=10);plt.xlabel('Total Claim Amount');plt.ylabel('frequency');plt.title('Total Claim Amount vs frequency') #histogram of Total Claim Amount is positively skewed.

# Plotting Boxplot to get the outliers from the dataset

plt.boxplot(autoinsu['Customer Lifetime Value'],vert=False);plt.xlabel('Customer Lifetime Value');plt.ylabel('frequency');plt.title('Customer Lifetime Value vs frequency') # There are outliers in the Customer Lifetime Value dataset

# We will replace the outliers by maximum and minimum method

IQR1 = autoinsu['Customer Lifetime Value'].quantile(0.75)-autoinsu['Customer Lifetime Value'].quantile(0.25)

IQR1

lowerlimit1 = autoinsu['Customer Lifetime Value'].quantile(0.25)-(1.5*IQR1)

lowerlimit1

upperlimit1 = autoinsu['Customer Lifetime Value'].quantile(0.75)+(1.5*IQR1)

upperlimit1

wins1a = autoinsu['Customer Lifetime Value'].quantile(0.05)

wins1a

wins1b = autoinsu['Customer Lifetime Value'].quantile(0.95)

wins1b

# to replace the outliers

autoinsu['Customer Lifetime Value'] = pd.DataFrame(np.where(autoinsu['Customer Lifetime Value']<lowerlimit1,wins1a,np.where(autoinsu['Customer Lifetime Value']>upperlimit1,wins1b,autoinsu['Customer Lifetime Value'])))

plt.boxplot(autoinsu['Customer Lifetime Value'],vert=False) #There are no outliers in the Balance dataset

plt.boxplot(autoinsu['Income'],vert=False);plt.xlabel('Income');plt.ylabel('frequency');plt.title('Income vs frequency') #There are no outliers in the income dataset

plt.boxplot(autoinsu['Monthly Premium Auto'],vert=False);plt.xlabel('Monthly Premium Auto');plt.ylabel('frequency');plt.title('Monthly Premium Auto vs frequency') #There are outliers in the Monthly Premium Auto dataset

# we will manually replace the  outliers value by maximum and minimum value

IQR2 = autoinsu['Monthly Premium Auto'].quantile(0.75)-autoinsu['Monthly Premium Auto'].quantile(0.25)

IQR2

lowerlimit2 = autoinsu['Monthly Premium Auto'].quantile(0.25)-(1.5*IQR2)

lowerlimit2

upperlimit2 = autoinsu['Monthly Premium Auto'].quantile(0.75)+(1.5*IQR2)

upperlimit2

wins2a = autoinsu['Monthly Premium Auto'].quantile(0.05)

wins2a

wins2b = autoinsu['Monthly Premium Auto'].quantile(0.95)

wins2b

# to replace the outliers

autoinsu['Monthly Premium Auto'] = pd.DataFrame(np.where(autoinsu['Monthly Premium Auto']<lowerlimit2,wins2a,np.where(autoinsu['Monthly Premium Auto']>upperlimit2,wins2b,autoinsu['Monthly Premium Auto'])))

plt.boxplot(autoinsu['Monthly Premium Auto'],vert=False) #There are no outliers in the Balance dataset

plt.boxplot(autoinsu['Months Since Last Claim'],vert=False);plt.xlabel('Months Since Last Claim');plt.ylabel('frequency');plt.title('Months Since Last Claim vs frequency') #There are no outliers in the Months Since Last Claim dataset.

plt.boxplot(autoinsu['Months Since Policy Inception'],vert=False);plt.xlabel('Months Since Policy Inception');plt.ylabel('frequency');plt.title('Months Since Policy Inception vs frequency')  #There are no outliers in the Months.Since.Policy.Inception dataset.

plt.boxplot(autoinsu['Number of Policies'],vert=False);plt.xlabel('Number of Policies');plt.ylabel('frequency');plt.title('Number of Policies vs frequency')  #There are outliers in the Number of Policies dataset.

# we will manually replace the  outliers value by maximum and minimum value

IQR3 = autoinsu['Number of Policies'].quantile(0.75)-autoinsu['Number of Policies'].quantile(0.25)

IQR3

lowerlimit3 = autoinsu['Number of Policies'].quantile(0.25)-(1.5*IQR3)

lowerlimit3

upperlimit3 = autoinsu['Number of Policies'].quantile(0.75)+(1.5*IQR3)

upperlimit3

wins3a = autoinsu['Number of Policies'].quantile(0.05)

wins3a

wins3b = autoinsu['Number of Policies'].quantile(0.95)

wins3b

# to replace the outliers

autoinsu['Number of Policies'] = pd.DataFrame(np.where(autoinsu['Number of Policies']<lowerlimit3,wins3a,np.where(autoinsu['Number of Policies']>upperlimit3,wins3b,autoinsu['Number of Policies'])))

plt.boxplot(autoinsu['Number of Policies'],vert=False) #There are no outliers in the Balance dataset

plt.boxplot(autoinsu['Total Claim Amount'],vert=False);plt.xlabel('Total Claim Amount');plt.ylabel('frequency');plt.title('Total Claim Amount vs frequency')  #There are outliers in the Total.Claim.Amount dataset.

# we will manually replace the  outliers value by maximum and minimum value

IQR4 = autoinsu['Total Claim Amount'].quantile(0.75)-autoinsu['Total Claim Amount'].quantile(0.25)

IQR4

lowerlimit4 = autoinsu['Total Claim Amount'].quantile(0.25)-(1.5*IQR4)

lowerlimit4

upperlimit4 = autoinsu['Total Claim Amount'].quantile(0.75)+(1.5*IQR4)

upperlimit4

wins4a = autoinsu['Total Claim Amount'].quantile(0.05)

wins4a

wins4b = autoinsu['Total Claim Amount'].quantile(0.95)

wins4b

# to replace the outliers

autoinsu['Total Claim Amount'] = pd.DataFrame(np.where(autoinsu['Total Claim Amount']<lowerlimit4,wins4a,np.where(autoinsu['Total Claim Amount']>upperlimit4,wins4b,autoinsu['Total Claim Amount'])))

plt.boxplot(autoinsu['Total Claim Amount'],vert=False) #There are no outliers in the Balance dataset

# Checking the normality of the dataset

import scipy.stats as stats

import pylab

stats.probplot(autoinsu['Customer Lifetime Value'],dist="norm",plot=pylab) # The Customer.Lifetime.Value data is non-normal data

# We have to apply transformation and check with it.

stats.probplot(np.log(autoinsu['Customer Lifetime Value']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(autoinsu['Customer Lifetime Value']),dist="norm",plot=pylab)

stats.probplot((1/(autoinsu['Customer Lifetime Value'])),dist="norm",plot=pylab)

# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data

stats.probplot(autoinsu['Income'],dist="norm",plot=pylab)  # The Income data is non-normal data.

# we have to apply transformation and check with it.

stats.probplot(np.log(autoinsu['Income']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(autoinsu['Income']),dist="norm",plot=pylab)

stats.probplot((1/(autoinsu['Income'])),dist="norm",plot=pylab)

# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data

stats.probplot(autoinsu['Monthly Premium Auto'],dist="norm",plot=pylab)    # The Monthly Premium Auto data is non-normal data

# we have to apply transformation and check with it.

stats.probplot(np.log(autoinsu['Monthly Premium Auto']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(autoinsu['Monthly Premium Auto']),dist="norm",plot=pylab)

stats.probplot((1/(autoinsu['Monthly Premium Auto'])),dist="norm",plot=pylab)

# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data

stats.probplot(autoinsu['Months Since Last Claim'],dist="norm",plot=pylab) # The Months Since Last Claim data is normal data.

stats.probplot(autoinsu['Months Since Policy Inception'],dist="norm",plot=pylab) # The Months.Since.Policy.Inception data is normal data.

stats.probplot(autoinsu['Number of Policies'],dist="norm",plot=pylab)  # The Number of Policies data is non-normal data.

# we have to apply transformation and check with it.

stats.probplot(np.log(autoinsu['Number of Policies']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(autoinsu['Number of Policies']),dist="norm",plot=pylab)

stats.probplot((1/(autoinsu['Number of Policies'])),dist="norm",plot=pylab)

# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data

stats.probplot(autoinsu['Total Claim Amount'],dist="norm",plot=pylab)  # The Total Claim Amount data is non-normal data.

# we have to apply transformation and check with it.

stats.probplot(np.log(autoinsu['Total Claim Amount']),dist="norm",plot=pylab)

stats.probplot(np.sqrt(autoinsu['Total Claim Amount']),dist="norm",plot=pylab)

# After applying sqrt transformation data is becoming normal data

autoinsu['Total Claim Amount'] = (np.sqrt(autoinsu['Total Claim Amount']))

# For the character column we have to apply dummy variable creation

autoinsu1 = autoinsu

autoinsu1.columns

autoinsu1 = autoinsu1.drop(['Customer','Effective To Date'],axis=1)

autodummy = pd.get_dummies(autoinsu1,columns=['State','Response','Coverage','Education','EmploymentStatus','Gender','Location Code','Marital Status','Policy Type','Policy','Renew Offer Type','Sales Channel','Vehicle Class','Vehicle Size'],drop_first=True)

# Here we are doing the dummy variable creation for the discrete data by ignoring first dummy variable.

# Here i am applying the technique if we have n model we need to create n-1 dummy variable.

# Checking fot zero variance in all columns

print(np.var(autodummy,axis=0))

# we have variance in count columns.

autodummy.describe()

# We will standardize the data to make it scale free

from sklearn.preprocessing import scale

standauto = scale(autodummy.iloc[:, 1:])

print(np.mean(standauto,axis=0))  # After standardizing we are getting mean as 0

print(np.var(standauto,axis=0))   # After standardizing we are getting variance as 1

# scree plot or elbow curve decide the k-value

from sklearn.cluster import	KMeans

twss=[]

k = list(range(2,9))

for i in k:

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(standauto)

    twss.append(kmeans.inertia_)

twss

# Scree plot or elbow curve decide the k-value

plt.plot(k,twss,'ro-');plt.xlabel("No of clusters");plt.ylabel("total_within_ss")

# slecting cluster

model = KMeans(n_clusters=3)

model.fit(standauto)

model.labels_

mb=pd.Series(model.labels_)

autoinsu['k_means_clust']=mb

autoinsu.head()

autoinsu = autoinsu.iloc[:, [0,24,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

autoinsu.head()

# Aggregate mean of each cluster

mean1 = autoinsu

mean1.columns

mean1 = mean1.drop(['Customer', 'k_means_clust', 'State', 'Response','Coverage', 'Education', 'Effective To Date', 'EmploymentStatus','Gender', 'Location Code', 'Marital Status','Policy Type','Policy', 'Renew Offer Type','Sales Channel', 'Vehicle Class','Vehicle Size'],axis=1)

mean1.iloc[:, 1:].groupby(autoinsu.k_means_clust).mean()

# we cannot apply mean to the character data, for only numeric data only we can apply, so we apply mode to character data

mode1 = autoinsu

mode1.columns

mode1 = mode1.drop(['Customer','Customer Lifetime Value', 'Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints','Number of Policies','Effective To Date','Total Claim Amount'],axis=1)

mode1.iloc[:, 1:].groupby(autoinsu.k_means_clust).agg(pd.Series.mode)

# creating a csv file 

autoinsu.to_csv("Kmeansautoinsu.csv", encoding = "utf-8")

import os

os.getcwd()

