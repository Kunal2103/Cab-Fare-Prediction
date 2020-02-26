##########     PROJECT:1 Cab-fare prediction     ##############

#Remove the elements
rm(list =ls())


#Set working directory
setwd("c:/rstudio")

#Check working directory
getwd()

#loading libaries foe operations
library(chron)
library(DMwR)
library(ggplot2)
library(scales)
library(gplots)
library(psych)
library(corrgram)
library(usdm)
library(rpart)
library(MASS)
library(randomForest)
library(RRF)
library(inTrees)



#loading datasets
cab = read.csv("train_cab.csv",header=T)


###################  Exploratory data analysis ###########################

#Getting the number of variables and obervation in the datasets
dim(cab)

#Structure of data
str(cab)

#converting fare_amount to numeric
cab$fare_amount = as.character(cab$fare_amount)
cab$fare_amount = as.numeric(cab$fare_amount)

#extracting different features from "pickup_datetime"

cab$pickup_date = as.Date(cab$pickup_datetime)
cab$days = days(cab$pickup_date)
cab$months = months(cab$pickup_date)
cab$weekdays = weekdays(cab$pickup_date)
cab$year = years(cab$pickup_date)
cab$time = substr(as.factor(cab$pickup_datetime),12,13) 

#converting datatypes of the new variables
cab$days = as.numeric(cab$days)
cab$year = as.numeric(cab$year)
cab$time = as.numeric(cab$time)

#converting factors to numeric for imputatioon
cab$months = as.factor(cab$months)
cab$weekdays = as.factor(cab$weekdays)
cab$months = as.numeric(cab$months)
cab$weekdays = as.numeric(cab$weekdays)

#removing unwanted features
cab$pickup_datetime = NULL
cab$pickup_date = NULL

#########################  DATA PREPROCESSING     ###########################################

### Missing Values Analysis ###

#replacing negative values of 'fare_amount' with zero
cab$fare_amount[ cab$fare_amount< 0 ] = 0

# find the count of missing values
sum(is.na(cab))

#replacing zeroes to NA
cab$fare_amount[cab$fare_amount == 0] = NA
cab$pickup_longitude[cab$pickup_longitude == 0] = NA
cab$pickup_latitude[cab$pickup_latitude == 0] = NA
cab$dropoff_longitude[cab$dropoff_longitude == 0] = NA
cab$dropoff_latitude[cab$dropoff_latitude == 0] = NA
cab$passenger_count[cab$passenger_count == 0] = NA
cab$months[cab$months == 0] = NA
cab$days[cab$days == 0] = NA
cab$weekdays[cab$weekdays == 0] = NA
cab$year[cab$year == 0] = NA

# find the count of missing values
sum(is.na(cab))


# create a dataframe with sum of missing values
missing_val = data.frame(apply(cab,2,function(x){sum(is.na(x))}))

# create a new column with variable names
missing_val$Columns = row.names(missing_val)

# renaming the column
names(missing_val)[1] =  "Missing_percentage"

# converting to percentage
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(cab)) * 100

# descending order of missing percentage
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL

# interchanging columns
missing_val = missing_val[,c(2,1)]

#observe a value & convert to NA
#cab[2,1]
#cab[2,1] = NA
# Actual value = 16.9
# Mean = 15.01
# Median = 8.5
# KNN = 7.68

#here mean method gives very close value to the actual value.so we will freeze mean method

#mean method
cab$fare_amount[is.na(cab$fare_amount)] = mean(cab$fare_amount, na.rm = T)

#mean method for remaining variables
cab$time[is.na(cab$time)] = mean(cab$time, na.rm = T)
cab$months[is.na(cab$months)] = mean(cab$months, na.rm = T)
cab$weekdays[is.na(cab$weekdays)] = mean(cab$weekdays, na.rm = T)
cab$year[is.na(cab$year)] = mean(cab$year, na.rm = T)
cab$passenger_count[is.na(cab$passenger_count)] = mean(cab$passenger_count, na.rm = T)
cab$pickup_longitude[is.na(cab$pickup_longitude)] = mean(cab$pickup_longitude, na.rm = T)
cab$pickup_latitude[is.na(cab$pickup_latitude)] = mean(cab$pickup_latitude, na.rm = T)
cab$dropoff_longitude[is.na(cab$dropoff_longitude)] = mean(cab$dropoff_longitude, na.rm = T)
cab$dropoff_latitude[is.na(cab$dropoff_latitude)] = mean(cab$dropoff_latitude, na.rm = T)
cab$days[is.na(cab$days)] = mean(cab$days,na.rm = T)

#median method
cab$fare_amount[is.na(cab$fare_amount)] = median(cab$fare_amount, na.rm = T)

#knn method
cab = knnImputation(cab, k = 5)

sum(is.na(cab))
cab = na.omit(cab)

#converting back some features to factors
cab$months = as.integer(cab$months)
cab$weekdays = as.integer(cab$weekdays)
cab$time = as.integer(cab$time)

cab$months = as.factor(cab$months)
cab$weekdays = as.factor(cab$weekdays)
cab$time = as.factor(cab$time)

rm(missing_val)

############### Outlier Analysis using Boxplot ###############

#boxplot
boxplot(cab$passenger_count,
        main = "Boxplot for passenger_count",
        xlab = "",
        ylab = "fare amount",
        col = "orange",
        border = "brown",
        horizontal = FALSE,
        notch = FALSE
)

numeric_index = sapply(cab,is.numeric)  #selecting only numeric
numeric_data = cab[,numeric_index]      #subset of numeric data
cnames = colnames(numeric_data)            #saving the column names of numeric data


#detect outliers
for(i in cnames){
  print(i)
  val = cab[,i][cab[,i] %in% boxplot.stats(cab[,i])$out]
  print(length(val))
  cab[,i][cab[,i] %in% val] = NA
}

#knn imputation for outliers
cab = knnImputation(cab, k = 5 )

#removing unwanted objects
rm(numeric_data)
rm(cnames)
rm(i)
rm(numeric_index)
rm(val)


############### Feature Selection ###############

numeric_index = sapply(cab,is.numeric)  #selecting only numeric
numeric_data = cab[,numeric_index]      #subset of numeric data


#correlation plot


corrgram(cab[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main= "Correlation plot")


#correlation matrix
cor_mat = cor(numeric_data)
cor_mat = round(cor_mat, 2)


#no eneed to remove any numeric feature


#removing unwanted objects
rm(cor_mat)
rm(numeric_data)
rm(numeric_index)


## Chi-squared Test of Independence
factor_index = sapply(cab,is.factor)
factor_data = cab[,factor_index]

for (i in 1:3)
  {
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data[,i])))
}

#no need to remove any factor features

rm(factor_data)
rm(factor_index)
rm(i)


############## Feature Scaling ###############


#normality check
hist(cab$passenger_count)              #not normally distributed


#to check range before normalisation
numeric_index = sapply(cab,is.numeric)  #selecting only numeric
numeric_data = cab[,numeric_index] #subset of numeric data
numeric_data$fare_amount = NULL
range(numeric_data)

#saving the column names of numeric data
cnames = colnames(numeric_data) 


#normalisation
for (i in cnames){
  print(i)
  cab[,i] = (cab[,i] - min(cab[,i])) / 
    (max(cab[,i] - min(cab[,i])))
}


#to check range after normalisation
numeric_index = sapply(cab,is.numeric)  #selecting only numeric
numeric_data = cab[,numeric_index] #subset of numeric data
numeric_data$fare_amount = NULL
range(numeric_data)


#removing unwanted objects
rm(numeric_data)
rm(cnames)
rm(i)
rm(numeric_index)


################## Modelling #######################

df = cab

#check & remove multicollinearity
vif(df[,-1])

numeric_index = sapply(df,is.numeric)  #selecting only numeric
numeric_data = df[,numeric_index]#subset of numeric data

vifcor(numeric_data[,-1], th = 0.9)



#divide into train & test
train_index = sample(1:nrow(df), 0.9 * nrow(df))
train = df[train_index,]
test = df[-train_index,]



###### Linear regression ######


#LR model
lm_model = lm(fare_amount ~., data = train)

summary(lm_model)

#prediction
predictions_lm = predict(lm_model, test[,2:11])

#evaluation
regr.eval(test[,1], predictions_lm, stats = c('mape','rmse'))


#Error[MAPE] = 39.69
#Accuracy = 60.31
#RMSE = 0.182


########## Decision Tree ###########

#DT model
DT = rpart(fare_amount ~ ., data = train, method = "anova")

#prediction
predictions_DT = predict(DT, test[,-1])

#evaluation
regr.eval(test[,1], predictions_DT, stats = c('mape','rmse'))


#Error[MAPE] = 36.49
#Accuracy = 63.51
#RMSE = 0.169


########## Random Forest ###########

#RF model
RF_model = randomForest(fare_amount ~ ., train, importance = TRUE, ntree = 100)

#prediction
RF_Predictions = predict(RF_model, test[,-1])

#evaluation 
regr.eval(test[,1], RF_Predictions, stats = c('mape','rmse'))

#Error[MAPE] = 27.25
#Accuracy = 72.75
#RMSE = 0.134

#we will freeze RANDOM FOREST model as it performs well.



################### prediction on large test data ###################

cab_test = read.csv("test.csv",header = T)


##### data exploration #####

str(cab_test)

#extracting the features from "pickup_datetime"

cab_test$pickup_date = as.Date(cab_test$pickup_datetime)
cab_test$days = days(cab_test$pickup_date)
cab_test$months = months(cab_test$pickup_date)
cab_test$weekdays = weekdays(cab_test$pickup_date)
cab_test$year = years(cab_test$pickup_date)
cab_test$time = substr(as.factor(cab_test$pickup_datetime),12,13) 

#converting the new variables
cab_test$days = as.numeric(cab_test$days)
cab_test$year = as.numeric(cab_test$year)

#converting factors to numeric for imputatioon
cab_test$time = as.numeric(cab_test$time)
cab_test$months = as.factor(cab_test$months)
cab_test$weekdays = as.factor(cab_test$weekdays)
cab_test$months = as.numeric(cab_test$months)
cab_test$weekdays = as.numeric(cab_test$weekdays)
cab_test$passenger_count = as.numeric(cab_test$passenger_count)

#removing unwanted features
cab_test$pickup_datetime = NULL
cab_test$pickup_date = NULL

################ data pre-processing ##################

############### Missing Value Analysis ###############


#find the count of missing values
sum(is.na(cab_test))

#replacing zeroes to NA
cab_test$pickup_longitude[cab_test$pickup_longitude == 0] = NA
cab_test$pickup_latitude[cab_test$pickup_latitude == 0] = NA
cab_test$dropoff_longitude[cab_test$dropoff_longitude == 0] = NA
cab_test$dropoff_latitude[cab_test$dropoff_latitude == 0] = NA
cab_test$passenger_count[cab_test$passenger_count == 0] = NA
cab_test$months[cab_test$months == 0] = NA
cab_test$days[cab_test$days == 0] = NA
cab_test$weekdays[cab_test$weekdays == 0] = NA
cab_test$year[cab_test$year == 0] = NA

#no missing vlue found


#converting back some features to factors
cab_test$months = as.factor(cab_test$months)
cab_test$weekdays = as.factor(cab_test$weekdays)
cab_test$time = as.factor(cab_test$time)


############## Feature Scaling ###############


#normality check
hist(cab_test$pickup_longitude)   #not normally distributed


#to check range before normalisation
numeric_index = sapply(cab_test,is.numeric)  #selecting only numeric
numeric_data = cab_test[,numeric_index] #subset of numeric data
range(numeric_data)

#saving the column names of numeric data
cnames = colnames(numeric_data) 



#normalisation
for (i in cnames){
  print(i)
  cab_test[,i] = (cab_test[,i] - min(cab_test[,i])) / 
    (max(cab_test[,i] - min(cab_test[,i])))
}


#to check range after normalisation
numeric_index = sapply(cab_test,is.numeric)  #selecting only numeric
numeric_data = cab_test[,numeric_index]      #subset of numeric data
range(numeric_data)


#removing unwanted objects
rm(numeric_data)
rm(cnames)
rm(i)
rm(numeric_index)


##### Prediction #######

#predictions
predictions_test = predict(RF_model, cab_test)

#save predictions as dataframe
predictions_test = as.data.frame(predictions_test)

#importing original test dataset
cab_results = read.csv("test.csv", header = T)

#columnbind target results with test data
cab_results = cbind(predictions_test, cab_results)

#renaming column
names(cab_results)[1] = "fare_amount_predicted"

#saving output in csv format
write.csv(cab_results, "Fare amount results r.csv", row.names = F)
