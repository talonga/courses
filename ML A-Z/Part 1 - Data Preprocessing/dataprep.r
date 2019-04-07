setwd('C:\\Users\\chaos\\Documents\\ML A-Z\\Part 1 - Data Preprocessing')

#Importing the dataset
data = read.csv('Data.csv')

#Taking care of missing data
data$Age = ifelse(is.na(data$Age),
                  ave(data$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                  data$Age)

data$Salary = ifelse(is.na(data$Salary),
                  ave(data$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                  data$Salary)

# Encoding categorical data
data$Country = factor(data$Country,
                        levels = c('France', 'Spain', 'Germany'), 
                        labels = c(1,2,3))

data$Purchased = factor(data$Purchased,
                      levels = c('No', 'Yes'),
                      labels = c(0,1))

# Splitting the dataset into the training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.8)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#feature scaling
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])