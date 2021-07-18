# 2
# Importing dataset
library(readr)
crimedata <- read.csv("C:/Users/WIN10/Desktop/LEARNING/crime_data.csv")
View(crimedata)
summary(crimedata)
# Performing EDA for the given dataset
# Checking for NA values
is.na(crimedata)
sum(is.na(crimedata))
# There are no missing values
# Plotting histogram for finding the Skewness
colnames(crimedata)
hist(crimedata$Murder,xlab = 'Murder',ylab = 'Frequency',col = 'green',border = 'red',breaks = 15)  # Histogram is positively skewed
hist(crimedata$Assault,xlab = 'Assault',ylab = 'Frequency',col = 'green',border = 'red',breaks = 15) # Histogram is normally skewed
hist(crimedata$UrbanPop,xlab = 'Urban Pop',ylab = 'Frequency',col = 'green',border = 'red',breaks = 15)  # Histogram is negatively skewed
hist(crimedata$Rape,xlab = 'Rape',ylab = 'frequency',col = 'green',border = 'red',breaks = 15)  # Histogram is negatively skewed
# Plotting boxplot for checking outliers
murderbox <- boxplot(crimedata$Murder,horizontal = T,xlab = 'Murder',ylab = 'Frequecy',main = "Murder vs Frequency",col = 'red',border = 'blue')
murderbox$out  # There are no outliers in the Murder dataset
assaultbox <- boxplot(crimedata$Assault,xlab = 'Assault',ylab = 'Frequecy',main = 'Assault vs Frequency',horizontal = T,col = 'red',border = 'blue')
assaultbox$out  # There are no outliers in assault dataset
urbanbox <- boxplot(crimedata$UrbanPop,horizontal = T,xlab = 'Urban Pop',ylab = 'Frequency',main = 'Urban Pop vs Frequency',col = 'red',border = 'blue')
urbanbox$out   # There are no outliers in urbanpop dataset
rapebox <- boxplot(crimedata$Rape,horizontal = T,xlab = 'Rape',ylab = 'Frequency', main = 'Rape vs Frequency',col = 'red',border = 'blue')
rapebox$out    # There are outliers in this dataset
# We will perform winsorization and remove the outliers
quant1 <- quantile(crimedata$Rape,probs = c(0.25,0.75))
quant1
wins1 <- quantile(crimedata$Rape,probs = c(0.05,0.95))
wins1
a1 <- 1.5*IQR(crimedata$Rape)
a1
b1 <- quant1[1] - a1
b1
c1 <- quant1[2] + a1
c1
# Replacing the outliers
crimedata$Rape[crimedata$Rape<b1] <- wins1[1]
crimedata$Rape[crimedata$Rape>c1] <- wins1[2]
# Checking the outliers greater than 95% limit
d <- boxplot(crimedata$Rape)
d$out  # The outliers are removed
# To check the normality of the dataset
qqnorm(crimedata$Murder)
qqline(crimedata$Murder)
# The Murder data is normal datae.
qqnorm(crimedata$Assault)
qqline(crimedata$Assault)
# The Assault data is normal data
qqnorm(crimedata$UrbanPop)
qqline(crimedata$UrbanPop)
# The urbanpop data is normal data
qqnorm(crimedata$Rape)
qqline(crimedata$Rape)
# The Rape data is normal data
# Since there is only continuous data there is no need for discritization
# Checking our dataset for zero variance column
# Check for variance on numerical values
apply(crimedata, 2, var) 
# Check for the columns having zero variance
which(apply(crimedata, 2, var)==0) 
# We have variance in all the columns.
# We have to make our data scale free and unit free so we have to do normalization
# normalize the data
# For normalization we define user defined function
norm <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}
normdata <- as.data.frame(lapply(crimedata[2:5],norm))
summary(normdata)
# Plotting the elbow curve
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
final <- data.frame(fit$cluster,crimedata)
aggregate(crimedata[, 2:5], by = list(final$cluster), FUN = mean)
# To get the output with having clustered group value column in it.
library(readr)
write_csv(final, "Kmeanscrime.csv")
getwd() #to get working directory
