# 1
# Importing the Dataset
library(readxl)
EastWestAirlines <- read_excel("C:/Users/WIN10/Desktop/LEARNING/EastWestAirlines.xlsx", sheet = "data")
View(EastWestAirlines)
summary(EastWestAirlines)
# Checking for NA values
is.na(EastWestAirlines)
sum(is.na(EastWestAirlines))
# Performing EDA for the dataset
# Plotting Histogram for finding skewness 
colnames(EastWestAirlines)  # For getting the column names from our dataset
hist(EastWestAirlines$Balance,xlab = 'Balance',ylab = 'Frequency',col = 'red',border = 'green',breaks = 30)   # Histogram is positively skewed
hist(EastWestAirlines$Bonus_miles,xlab = 'Bonus Miles',ylab = 'Frequency',col = 'red',border = 'green',breaks = 30) # Histogram is positively skewed
hist(EastWestAirlines$Bonus_trans,xlab = 'Bonus Trans',ylab = 'Frequency',col = 'red',border = 'green',breaks = 30) # Histogram is positively skewed
hist(EastWestAirlines$Days_since_enroll,xlab = 'Days Since Enroll',ylab = 'Frequency',col = 'red',border = 'green',breaks = 30) # Histogram is normally skewed
# Plotting Box plot for identifying outliers
bal <- boxplot(EastWestAirlines$Balance,horizontal = T,xlab = 'Balance',ylab = 'Frequency',main = 'Balance vs Frequency',col = 'yellow',border = 'green') # Outliers are existing
bal$out
# Replacing Ourtliers by Winsorization method
quant1 <- quantile(EastWestAirlines$Balance,probs = c(0.25,0.75))
quant1
wins1 <- quantile(EastWestAirlines$Balance,probs = c(0.05,0.95))
wins1
a1 <- 1.5*IQR(EastWestAirlines$Balance)
a1
b1 <- quant1[1] - a1
b1
c1 <- quant1[2] + a1
c1
# Replacing the outliers 
EastWestAirlines$Balance[EastWestAirlines$Balance<b1] <- wins1[1]
EastWestAirlines$Balance[EastWestAirlines$Balance>c1] <- wins1[2]
# Checking the outliers greater than the 95% limit value
d1 <- boxplot(EastWestAirlines$Balance)
d1$out
bonmiles <- boxplot(EastWestAirlines$Bonus_miles,horizontal = T,xlab = 'Bonus Miles',ylab = 'Frequency',main = 'Bonus Miles vs Frequency',col = 'yellow',border = 'green')  # Outliers are existing in our dataset
bonmiles$out
# Replacing the outliers by Winsorization
quant2 <- quantile(EastWestAirlines$Bonus_miles,probs = c(0.25,0.75))
quant2
wins2 <- quantile(EastWestAirlines$Bonus_miles,probs = c(0.05,0.95))
wins2
a2 <- 1.5*IQR(EastWestAirlines$Bonus_miles)
a2
b2 <- quant2[1] - a2
b2
c2 <- quant2[2] + a2
c2
# Replacing the outliers
EastWestAirlines$Bonus_miles[EastWestAirlines$Bonus_miles<b2] <- wins2[1]
EastWestAirlines$Bonus_miles[EastWestAirlines$Bonus_miles>c2] <- wins2[2]
# Checking the outliers greater than 95% limit value
d2 <- boxplot(EastWestAirlines$Bonus_miles)
d2$out
bontrans <- boxplot(EastWestAirlines$Bonus_trans,horizontal = T,xlab = 'Bonus Trans',ylab = 'Frequency',main = 'Bonus Trans vs Frequency',col = 'yellow',border = 'green')
bontrans$out
# Replacing outliers by Winsorizaton method
quant3 <- quantile(EastWestAirlines$Bonus_trans,probs = c(0.25,0.75))
quant3
wins3 <- quantile(EastWestAirlines$Bonus_trans,probs = c(0.05,0.95))
wins3
a3 <- 1.5*IQR(EastWestAirlines$Bonus_trans)
a3
b3 <- quant3[1] - a3
b3
c3 <- quant3[2] + a3
c3
# Replacing the outliers
EastWestAirlines$Bonus_trans[EastWestAirlines$Bonus_trans<b3] <- wins3[1]
EastWestAirlines$Bonus_trans[EastWestAirlines$Bonus_trans>c3] <- wins3[2]
# Checking outliers greater than 95% limit value
d3 <- boxplot(EastWestAirlines$Bonus_trans)
d3$out
dse <- boxplot(EastWestAirlines$Days_since_enroll,horizontal = T,xlab = 'Days Since Enroll',ylab = 'Frequency',main = 'Days since Enroll vs Frequency',col = 'yellow',border = 'green')
dse$out  # There are no outliers in this column
# For checking the Normality of dataset
qqnorm(EastWestAirlines$Balance)
qqline(EastWestAirlines$Balance)
# Therefore Balance dataset is non normal
# Therefore we have will do transformation 
# We will proceed with log transformation
qqnorm(log(EastWestAirlines$Balance),ylim = c(0,20))
qqline(log(EastWestAirlines$Balance))
# After applying log transformation the data is normal
qqnorm(EastWestAirlines$Bonus_miles)
qqline(EastWestAirlines$Bonus_miles)
# The data is non normal, therefore we will apply log transformation
qqnorm(log(EastWestAirlines$Bonus_miles),ylim = c(1,10))
qqline(log(EastWestAirlines$Bonus_miles))
# Therefore the data is normal
qqnorm(EastWestAirlines$Bonus_trans)
qqline(EastWestAirlines$Bonus_trans)
# The data is non normal, therefore we will apply transformation
qqnorm(log(EastWestAirlines$Bonus_trans),ylim = c(1,10))
qqline(log(EastWestAirlines$Bonus_trans))
# The data is still not normal
qqnorm(sqrt(EastWestAirlines$Bonus_trans))
qqline(sqrt(EastWestAirlines$Bonus_trans))
# The data is not yet normal so we can proceed with non normal data
qqnorm(EastWestAirlines$Days_since_enroll)
qqline(EastWestAirlines$Days_since_enroll)
# The data is normal, so there is no need for transformation
# Checking the columns for zero variance, if there is a zero column we can ignore it
# Check for variance on numerical values
apply(EastWestAirlines,2,var)
# Check for the columns having zero variance
which(apply(EastWestAirlines,2,var)==0)
# Therefore we have variance in all columns
# Normalizing the data
norm <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}
# Applying normalization for dataset
normdata <- as.data.frame(lapply(EastWestAirlines[2:11],norm))
summary(normdata)
# Plotting the elbow curve
twss <- NULL
for (i in 2:8) {
  twss <- c(twss,kmeans(normdata,centers = i)$tot.withinss)
}
twss
# Plotting elbow curve in scree plot
plot(2:8,twss,type = "b",xlab = "Number of clusters",ylab = "within groups sum of squares")
title(sub = "Airlines K Means")
# Cluster Solution
fit <- kmeans(normdata, 3) 
str(fit)
fit$cluster
final <- data.frame(fit$cluster, EastWestAirlines) # Append cluster membership
aggregate(EastWestAirlines[, 2:12], by = list(fit$cluster), FUN = mean)
# To get the output with having clustered group value column in it.
library(readr)
write_csv(final, "Kmeanscrime.csv")
getwd() #to get working directory
