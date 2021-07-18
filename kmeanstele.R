# 3
library(readxl)
telecom <- read_excel("C:/Users/WIN10/Desktop/LEARNING/Telco_customer_churn.xlsx")
View(telecom)
# Performing EDA on our dataset
# Checking for NA values in our dataset
table(is.na(telecom)) 
# There are no NA values present in our dataset
# Plotting histogram to get the skeweness of the dataset
colnames(telecom)
attach(telecom)
hist(`Tenure in Months`,xlab = 'Tenure in Months',ylab= 'frequency',col= 'blue',border = 'red',breaks = 45) #histogram of Tenure in months is normally skewed
hist(`Avg Monthly Long Distance Charges`,xlab = 'Avg Monthly Long Distance Charges',ylab= 'frequency',col= 'blue',border = 'red',breaks = 45) #histogram of Avg Monthly Long Distance Charges is normally skewed
hist(`Avg Monthly GB Download`,xlab = 'Avg Monthly GB Download',ylab= 'frequency',col= 'blue',border = 'red',breaks = 30) #histogram of Avg Monthly GB Download is positively skewed
hist(`Monthly Charge`,xlab = 'Monthly Charge',ylab= 'frequency',col= 'blue',border = 'red',breaks = 30) #histogram of Monthly Charge is Normally skewed
hist(`Total Charges`,xlab = 'Total Charges',ylab= 'frequency',col= 'blue',border = 'red',breaks = 30) #histogram of Total Charges is positively skewed
hist(`Total Long Distance Charges`,xlab = 'Total Long Distance Charges',ylab= 'frequency',col= 'blue',border = 'red',breaks = 40) #histogram of Long distance charges is positively skewed
hist(`Total Revenue`,xlab = 'Total Revenue',ylab= 'frequency',col= 'blue',border = 'red',breaks = 40) #histogram of Total revenue is positively skewed
# Plotting boxplot to get the outliers from the dataset
tim <- boxplot(`Tenure in Months`,horizontal = T,xlab='Tenure in Months',ylab='frequency',main='Tenure in Months vs frequency',col = 'blue',border='red') 
tim$out #There are no outliers in the Tenure in Months dataset
aml <- boxplot(`Avg Monthly Long Distance Charges`,horizontal = T,xlab='Avg Monthly Long Distance Charges',ylab='frequency',main='Avg Monthly Long Distance Charges vs frequency',col = 'blue',border='red') 
aml$out #There are no outliers in the Avg Monthly Long Distance Charges dataset
amd <- boxplot(`Avg Monthly GB Download`,horizontal = T,xlab='Avg Monthly GB Download',ylab='frequency',main='Avg Monthly GB Download vs frequency',col = 'blue',border='red') 
amd$out #There are outliers in the Avg Monthly GB Download dataset
# Replacing the outliers by winsorization
qunt1 <- quantile(`Avg Monthly GB Download`,probs = c(0.25,0.75))
qunt1
wins1 <- quantile(`Avg Monthly GB Download`,probs = c(0.05,0.95))
wins1
a1 <- 1.5*IQR(`Avg Monthly GB Download`)
a1
b1 <- qunt1[1]-a1
b1
c1 <- qunt1[2]+a1
c1
# Replace the outliers with winsorization method
telecom$`Avg Monthly GB Download`[telecom$`Avg Monthly GB Download`<b1] <- wins1[1]
telecom$`Avg Monthly GB Download`[telecom$`Avg Monthly GB Download`>c1] <- wins1[2]
# To check with outliers greater than 95% of the limit value 
d1 <- boxplot(telecom$`Avg Monthly GB Download`)
d1$out #There are no outliers in the Avg Monthly GB Download dataset.
mc <- boxplot(`Monthly Charge`,horizontal = T,xlab='Monthly Charge',ylab='frequency',main='Monthly Charge vs frequency',col = 'blue',border='red') 
mc$out #There are no outliers in the Monthly Charge dataset
tc <- boxplot(`Total Charges`,horizontal = T,xlab='Total Charges',ylab='frequency',main='Total Charges vs frequency',col = 'blue',border='red') 
tc$out #There are no outliers in the Total Charges dataset
tld <- boxplot(`Total Long Distance Charges`,horizontal = T,xlab='Total Long Distance Charges',ylab='frequency',main='Total Long Distance Charges vs frequency',col = 'blue',border='red') 
tld$out #There are outliers in the Total Long Distance Charges dataset
# Replace the outliers value with selected minimum and maximum value-winsorization
qunt2 <- quantile(`Total Long Distance Charges`,probs = c(0.25,0.75))
qunt2
wins2 <- quantile(`Total Long Distance Charges`,probs = c(0.05,0.95))
wins2
a2 <- 1.5*IQR(`Total Long Distance Charges`)
a2
b2 <- qunt2[1]-a2
b2
c2 <- qunt2[2]+a2
c2
# Replace the outliers with winsorization method
telecom$`Total Long Distance Charges`[telecom$`Total Long Distance Charges`<b2] <- wins2[1]
telecom$`Total Long Distance Charges`[telecom$`Total Long Distance Charges`>c2] <- wins2[2]
# To check with outliers greater than 95% of the limit value 
d2 <- boxplot(telecom$`Total Long Distance Charges`)
d2$out #There are no outliers in the Total Long Distance Charges dataset.
tr <- boxplot(`Total Revenue`,horizontal = T,xlab='Total Revenue',ylab='frequency',main='Total Revenue vs frequency',col = 'blue',border='red') 
tr$out #There are outliers in the Total Revenue dataset
# Replace the outliers value with winsorization
qunt3 <- quantile(`Total Revenue`,probs = c(0.25,0.75))
qunt3
wins3 <- quantile(`Total Revenue`,probs = c(0.05,0.99))
wins3
a3 <- 1.5*IQR(`Total Revenue`)
a3
b3 <- qunt3[1]-a3
b3
c3 <- qunt3[2]+a3
c3
# Replace the outliers with winsorization method
telecom$`Total Revenue`[telecom$`Total Revenue`<b3] <- wins3[1]
telecom$`Total Revenue`[telecom$`Total Revenue`>c3] <- wins3[2]
# To check with outliers greater than 95% of the limit value 
d3 <- boxplot(telecom$`Total Revenue`)
d3$out #There are no outliers in the Total Revenue dataset.
# To check the normality of the dataset
qqnorm(`Tenure in Months`)
qqline(`Tenure in Months`)
# The data is normal data
qqnorm(`Avg Monthly Long Distance Charges`)
qqline(`Avg Monthly Long Distance Charges`)
# The data is normal data
qqnorm(`Avg Monthly GB Download`)
qqline(`Avg Monthly GB Download`)
# The data is non-normal data
# we have to apply transformation and check with it.
qqnorm(log(`Avg Monthly GB Download`),ylim = c(0,5))
qqline(log(`Avg Monthly GB Download`))
qqnorm(sqrt(`Avg Monthly GB Download`))
qqline(sqrt(`Avg Monthly GB Download`))
qqnorm(1/(`Avg Monthly GB Download`),ylim = c(0,1))
qqline(1/(`Avg Monthly GB Download`))
# Even after applying transformation the data is still non-normal so we will proceed it as non-normal data
qqnorm(`Monthly Charge`)
qqline(`Monthly Charge`)
# The data is non-normal data
qqnorm(log(`Monthly Charge`),ylim = c(0,5))
qqline(log(`Monthly Charge`))
qqnorm(sqrt(`Monthly Charge`))
qqline(sqrt(`Monthly Charge`))
qqnorm(1/(`Monthly Charge`),ylim = c(0,0.1))
qqline(1/(`Monthly Charge`))
# even after applying transformation the data is still non-normal so we will proceed it as non-normal data
qqnorm(`Total Charges`)
qqline(`Total Charges`)
# The data is non-normal data
qqnorm(log(`Total Charges`),ylim = c(2,5))
qqline(log(`Total Charges`))
qqnorm(sqrt(`Total Charges`))
qqline(sqrt(`Total Charges`))
# After applying sqrt transformation the data is converting to normal data, so we will apply sqrt transformation the total charges column
telecom$`Total Charges` <- (sqrt(telecom$`Total Charges`))
qqnorm(`Total Long Distance Charges`)
qqline(`Total Long Distance Charges`)
# The data is non-normal data
qqnorm(log(`Total Long Distance Charges`),ylim = c(0,5))
qqline(log(`Total Long Distance Charges`))
qqnorm(sqrt(`Total Long Distance Charges`))
qqline(sqrt(`Total Long Distance Charges`))
# After applying sqrt transformation the data is converting to normal data, so we will apply sqrt transformation the total charges column
telecom$`Total Long Distance Charges` <- (sqrt(telecom$`Total Long Distance Charges`))
qqnorm(`Total Revenue`)
qqline(`Total Revenue`)
# The data is non-normal data
qqnorm(log(`Total Revenue`),ylim = c(2,10))
qqline(log(`Total Revenue`))
qqnorm(sqrt(`Total Revenue`))
qqline(sqrt(`Total Revenue`))
# After applying sqrt transformation the data is converting to normal data.
telecom$`Total Revenue` <- (sqrt(telecom$`Total Revenue`))
# For the character column we have to apply dummy variable creation
library(fastDummies)
teledummy <- dummy_cols(telecom,select_columns = c('Quarter','Referred a Friend','Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security','Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract','Paperless Billing','Payment Method'),remove_first_dummy = T,remove_most_frequent_dummy = F,remove_selected_columns = T)
# Here we are doing the dummy variable creation for the discrete data by ignoring first dummy variable.
# Here i am applying the technique if we have n model we need to create n-1 dummy variable.
# check for variance on numerical values
apply(teledummy, 2, var) 
# Check for the columns having zero variance
which(apply(teledummy, 2, var)==0) 
# we have zero variance in the column count and quarter so it won't give an impact to our dataset so we can ignore it.
# we are dropping count and quarter column
teledummy <- teledummy[,c(-2,-13)]
summary(teledummy)
# standardize the data and make it as scale,free and unit free.
normdata <- as.data.frame(scale(teledummy[,2:36]))
summary(normdata)
# after standardizing we are getting mean as 0
apply(normdata,2,var)
# after standardizing we are getting variance as 1
# Plotting the elbow curve
table(is.na(normdata))
summary(normdata)
twss<-NULL
for (i in 2:8){
  twss<-c(twss,kmeans(normdata,centers = i)$tot.withinss)
}
twss
# to visualize elbow curve in the screeplot
plot(2:8,twss,type = "b",xlab = "Number of clusters",ylab = "Within sum of square")
title("Kmeans clustering scree plot")
# Clustering solution
fit <- kmeans(normdata,3)
str(fit)
final <- data.frame(fit$cluster,telecom)
aggregate(telecom[, c(6,9,13,25,26,27,28,29,30)], by = list(fit$cluster), FUN = mean)
# We cannot apply mean to the character data, for only numeric data only we can apply, so we apply mode to character data
Mode <- function(x){
  a = unique(x) # x is a vector
  return(a[which.max(tabulate(match(x, a)))])
}
# we apply mode to character data 
aggregate(telecom[, c(-6,-9,-13,-25,-26,-27,-28,-29,-30)], by = list(fit$cluster), FUN = Mode)
# To get the output with having clustered group value column in it.
library(readr)
write_csv(final, "Kmeans_Tele_customer_churn.csv")
getwd() #to get working directory
