# 4
library(readr)
autodata <- read.csv("C:/Users/WIN10/Desktop/LEARNING/Autoinsurance.csv")
View(autodata)
summary(autodata)
# Performing EDA for the given dataset
sum(is.na(autodata))  # There are no NA values in our dataset
# Plotting histogram for getting skewness
hist(autodata$Customer.Lifetime.Value,xlab = 'Customer Lifetime Value',ylab = 'Frequency',col = 'red',border = 'blue',breaks = 15)  # Histogram is positively skewed
hist(autodata$Income,xlab = 'Income',ylab = 'Frequency',col = 'red',border = 'blue',breaks = 15)  # Histogram is positively skewed
hist(autodata$Monthly.Premium.Auto,xlab = 'Monthly premium auto',ylab = 'Frequency',col = 'red',border = 'blue',breaks = 15)  # Histogram is positively skewed
hist(autodata$Months.Since.Last.Claim,xlab = 'Monthly since last claim',ylab = 'Frequency',col = 'red',border = 'blue',breaks = 15)  # Histogram is positively skewed
hist(autodata$Months.Since.Policy.Inception,xlab = 'Months since policy inception',ylab = 'Frequency',col = 'red',border = 'blue',breaks = 15)  # Histogram is normally skewed
hist(autodata$Number.of.Policies,xlab = 'Number of policies',ylab = 'Frequency',col = 'red',border = 'blue',breaks = 15)  # Histogram is positively skewed
hist(autodata$Total.Claim.Amount,xlab = 'Total claim amount',ylab = 'Frequency',col = 'red',border = 'blue',breaks = 15)  # Histogram is positively skewed
# Boxplot to get the outliers from the dataset
clv <- boxplot(autodata$Customer.Lifetime.Value,horizontal = T,xlab='Customer.Lifetime.Value',ylab='frequency',main='Customer.Lifetime.Value vs frequency',col = 'blue',border='red') 
clv$out # There are outliers in the Customer.Lifetime.Value dataset
# Replace the outliers value with selected minimum and maximum value-winsorization
qunt1 <- quantile(autodata$Customer.Lifetime.Value,probs = c(0.25,0.75))
qunt1
wins1 <- quantile(autodata$Customer.Lifetime.Value,probs = c(0.05,0.95))
wins1
a1 <- 1.5*IQR(autodata$Customer.Lifetime.Value)
a1
b1 <- qunt1[1]-a1
b1
c1 <- qunt1[2]+a1
c1
# To replace the outliers with winsorization method
autodata$Customer.Lifetime.Value[autodata$Customer.Lifetime.Value<b1] <- wins1[1]
autodata$Customer.Lifetime.Value[autodata$Customer.Lifetime.Value>c1] <- wins1[2]
# To check with outliers greater than 95% of the limit value 
d1 <- boxplot(autodata$Customer.Lifetime.Value)
d1$out # There are no outliers in the Avg Monthly GB Download dataset.
incomebox <- boxplot(autodata$Income,horizontal = T,xlab='Income',ylab='frequency',main='Income vs frequency',col = 'blue',border='red') 
incomebox$out # There are no outliers in the income dataset
mpa <- boxplot(autodata$Monthly.Premium.Auto,horizontal = T,xlab='Monthly.Premium.Auto',ylab='frequency',main='Monthly.Premium.Auto vs frequency',col = 'blue',border='red') 
mpa$out # There are outliers in the Monthly.Premium.Auto dataset
# Replace the outliers value with selected minimum and maximum value-winsorization
qunt2 <- quantile(autodata$Monthly.Premium.Auto,probs = c(0.25,0.75))
qunt2
wins2 <- quantile(autodata$Monthly.Premium.Auto,probs = c(0.05,0.95))
wins2
a2 <- 1.5*IQR(autodata$Monthly.Premium.Auto)
a2
b2 <- qunt2[1]-a2
b2
c2 <- qunt2[2]+a2
c2
# Replacing the outliers with winsorization method
autodata$Monthly.Premium.Auto[autodata$Monthly.Premium.Auto<b2] <- wins2[1]
autodata$Monthly.Premium.Auto[autodata$Monthly.Premium.Auto>c2] <- wins2[2]
# To check with outliers greater than 95% of the limit value 
d2 <- boxplot(autodata$Monthly.Premium.Auto)
d2$out #There are no outliers in the Monthly.Premium.Auto dataset.
mlc <- boxplot(autodata$Monthly.Premium.Auto,horizontal = T,xlab='Months.Since.Last.Claim',ylab='frequency',main='Months.Since.Last.Claim vs frequency',col = 'blue',border='red') 
mlc$out #There are no outliers in the Months.Since.Last.Claim dataset
msp <- boxplot(autodata$Months.Since.Policy.Inception,horizontal = T,xlab='Months.Since.Last.Claim',ylab='frequency',main='Months.Since.Last.Claim vs frequency',col = 'blue',border='red') 
msp$out #There are no outliers in the Months.Since.Policy.Inception dataset
nop <- boxplot(autodata$Number.of.Policies,horizontal = T,xlab='Months.Since.Last.Claim',ylab='frequency',main='Months.Since.Last.Claim vs frequency',col = 'blue',border='red') 
nop$out #There are outliers in the Number.of.Policies dataset
# Replace the outliers value with selected minimum and maximum value-winsorization
qunt3 <- quantile(autodata$Number.of.Policies,probs = c(0.25,0.75))
qunt3
wins3 <- quantile(autodata$Number.of.Policies,probs = c(0.05,0.95))
wins3
a3 <- 1.5*IQR(autodata$Number.of.Policies)
a3
b3 <- qunt3[1]-a3
b3
c3 <- qunt3[2]+a3
c3
# To replace the outliers with winsorization method
autodata$Number.of.Policies[autodata$Number.of.Policies<b3] <- wins3[1]
autodata$Number.of.Policies[autodata$Number.of.Policies>c3] <- wins3[2]
# to check with outliers greater than 95% of the limit value 
d3 <- boxplot(autodata$Number.of.Policies)
d3$out #There are no outliers in the Number.of.Policies dataset.
tca <- boxplot(autodata$Total.Claim.Amount,horizontal = T,xlab='Months.Since.Last.Claim',ylab='frequency',main='Months.Since.Last.Claim vs frequency',col = 'blue',border='red') 
tca$out #There are outliers in the Total.Claim.Amount dataset.
# Replace the outliers value with selected minimum and maximum value-winsorization
qunt4 <- quantile(autodata$Total.Claim.Amount,probs = c(0.25,0.75))
qunt4
wins4 <- quantile(autodata$Total.Claim.Amount,probs = c(0.05,0.95))
wins4
a4 <- 1.5*IQR(autodata$Total.Claim.Amount)
a4
b4 <- qunt4[1]-a4
b4
c4 <- qunt4[2]+a4
c4
#to replace the outliers with winsorization method
autodata$Total.Claim.Amount[autodata$Total.Claim.Amount<b4] <- wins4[1]
autodata$Total.Claim.Amount[autodata$Total.Claim.Amount>c4] <- wins4[2]
#to check with outliers greater than 95% of the limit value 
d4 <- boxplot(autodata$Total.Claim.Amount)
d4$out #There are no outliers in the Number.of.Policies dataset.
# To check the normality of the dataset
qqnorm(autodata$Customer.Lifetime.Value)
qqline(autodata$Customer.Lifetime.Value)
# The Customer.Lifetime.Value data is non-normal data
# applying sqrt log and reciprocal transformation
qqnorm(log(autodata$Customer.Lifetime.Value))
qqline(log(autodata$Customer.Lifetime.Value))
qqnorm(sqrt(autodata$Customer.Lifetime.Value))
qqline(sqrt(autodata$Customer.Lifetime.Value))
qqnorm(1/(autodata$Customer.Lifetime.Value))
qqline(1/(autodata$Customer.Lifetime.Value))
# even after applying transformation the data is still non-normal so we will proceed it as non-normal data
qqnorm(autodata$Income)
qqline(autodata$Income)
# The Income data is non-normal data
#applying sqrt log and reciprocal transformation
qqnorm(log(autodata$Income),ylim = c(0,20))
qqline(log(autodata$Income))
qqnorm(sqrt(autodata$Income))
qqline(sqrt(autodata$Income))
qqnorm(1/(autodata$Income),ylim = c(0,0.5))
qqline(1/(autodata$Income))
# even after applying transformation the data is still non-normal so we will proceed it as non-normal data
qqnorm(autodata$Monthly.Premium.Auto)
qqline(autodata$Monthly.Premium.Auto)
# The Monthly.Premium.Auto data is non-normal data
#applying sqrt log and reciprocal transformation
qqnorm(log(autodata$Monthly.Premium.Auto),ylim = c(0,20))
qqline(log(autodata$Monthly.Premium.Auto))
# after applying log transformation the data is converting to normal data.
autodata$Monthly.Premium.Auto <- (log(autodata$Monthly.Premium.Auto))
qqnorm(autodata$Months.Since.Last.Claim)
qqline(autodata$Months.Since.Last.Claim)
# The Months.Since.Last.Claim data is normal data.
qqnorm(autodata$Months.Since.Policy.Inception)
qqline(autodata$Months.Since.Policy.Inception)
# The Months.Since.Policy.Inception data is normal data.
qqnorm(autodata$Number.of.Policies)
qqline(autodata$Number.of.Policies)
#The Number.of.Policies data is non-normal data.
#applying sqrt log and reciprocal transformation
qqnorm(log(autodata$Number.of.Policies),ylim = c(0,5))
qqline(log(autodata$Number.of.Policies))
qqnorm(sqrt(autodata$Number.of.Policies))
qqline(sqrt(autodata$Number.of.Policies))
qqnorm(1/(autodata$Number.of.Policies),ylim = c(0,0.5))
qqline(1/(autodata$Number.of.Policies))
# even after applying transformation the data is still non-normal so we will proceed it as non-normal data
qqnorm(autodata$Total.Claim.Amount)
qqline(autodata$Total.Claim.Amount)
#The Total.Claim.Amount data is non-normal data.
#applying sqrt log and reciprocal transformation
qqnorm(log(autodata$Total.Claim.Amount),ylim = c(0,20))
qqline(log(autodata$Total.Claim.Amount))
#after applying log transformation the data is converting to normal data.
autodata$Total.Claim.Amount <- (log(autodata$Total.Claim.Amount))
# For the character column we have to apply dummy variable creation
library(fastDummies)
autodata1 <- autodata
# we have to drop nominal data column
autodata1 <- autodata1[,c(-1,-7)]
colnames(autodata1)
autodummy <- dummy_cols(autodata1,select_columns = c('State','Response','Coverage','Education','EmploymentStatus','Gender','Location.Code','Marital.Status','Policy.Type','Policy','Renew.Offer.Type','Sales.Channel','Vehicle.Class','Vehicle.Size'),remove_first_dummy = T,remove_most_frequent_dummy = F,remove_selected_columns = T)
# Here we are doing the dummy variable creation for the discrete data by ignoring first dummy variable.
# Here i am applying the technique if we have n model we need to create n-1 dummy variable.
# check for variance on numerical values
apply(autodummy, 2, var) 
# Check for the columns having zero variance
which(apply(autodummy, 2, var)==0)
# we have variance in all the column of our dataset.
summary(autodummy)
# Normalizing the data
normautodata <- as.data.frame(scale(autodummy[,1:51]))
summary(normautodata)
# Plotting the elbow curve
twss<-NULL
for (i in 2:8){
  twss<-c(twss,kmeans(normautodata,centers = i)$tot.withinss)
}
twss
# to visualize elbow curve in the screeplot
plot(2:8,twss,type = "b",xlab = "Number of clusters",ylab = "Within sum of square")
title("Kmeans clustering scree plot")
# Clustering solution
fit <- kmeans(normautodata,3)
str(fit)
final <- data.frame(fit$cluster,autodata1)
a <- autodata1
a <- a[,-c(2,8,11,12,13,14,15,20)]
b <- autodata1
b <- b[,-c(1,3,4,5,6,7,9,10,16,17,18,19,21,22)]
b
aggregate(b[,1:8],by=list(fit$cluster),FUN = mean)
# we cannot apply mean to the character data, for only numeric data only we can apply, so we apply mode to character data
Mode <- function(x){
  a = unique(x) 
  return(a[which.max(tabulate(match(x, a)))])
}
# we apply mode to character data 
aggregate(autodata1[, c(-1,-3,-7,-10,-13,-14,-15,-16,-17,-22)], by = list(fit$cluster), FUN = Mode)
# To get the output with having clustered group value column in it.
library(readr)
write_csv(final, "Kmeansauto.csv")
getwd() #to get working directory
